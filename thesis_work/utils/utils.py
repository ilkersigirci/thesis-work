import io
import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples

from thesis_work.utils.initialization import check_initialization_params

logger = logging.getLogger(__name__)


def thread_pool_info():
    import numpy  # noqa: F401
    from threadpoolctl import threadpool_info

    logger.info(threadpool_info())


def limit_numpy_num_threads(num_threads: Optional[int] = None) -> None:
    """Limits total number of threads used by numpy.

    Need to be set before importing numpy. This is alternative for `threadpool_limits`.
    Try using `threadpool_limits` method first.
    """
    if num_threads is None:
        return

    import os

    num_threads = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads


def initialize_logger(logging_level: int = logging.INFO):
    logging.basicConfig()
    logging.getLogger().setLevel(logging_level)


def ignore_warnings(log_level: int = logging.ERROR):
    # Silence transformers warnings
    logging.getLogger("transformers.modeling_utils").setLevel(log_level)
    # from transformers import logging
    # logging.set_verbosity_error()

    # Silence RDKit warnings
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)


def check_device(device: str = "cuda"):
    check_initialization_params(attr=device, accepted_list=["cpu", "cuda"])

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda device requires GPU")


def log_plotly_figure(figure, name: str, method="static"):
    check_initialization_params(attr=method, accepted_list=["static", "dynamic"])

    if method == "static":
        fig_bytes = figure.to_image(format="png", engine="kaleido")
        img = Image.open(io.BytesIO(fig_bytes))
        img = np.asarray(img)

        wandb.log({name: wandb.Image(img)})

    elif method == "dynamic":
        table = wandb.Table(columns=["plotly_figure"])
        path_to_plotly_html = "./plotly_figure.html"

        # NOTE: Doesn't work because of utf-8 encoding problem in wandb
        # figure.write_html(path_to_plotly_html, auto_play=False)
        figure_html = figure.to_html(path_to_plotly_html, auto_play=False)

        table.add_data(wandb.Html(figure_html, inject=True))
        wandb.log({name: table})


def is_valid_smiles(smiles: str) -> bool:
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            return True
        else:
            return False
    except Exception:
        return False


def get_ecfp_descriptor(
    smiles_str: str, radius: int = 2, nBits: int = 2048, return_type="numpy"
) -> Union[np.array, ExplicitBitVect]:
    if not is_valid_smiles(smiles_str):
        raise ValueError("Invalid SMILES string")

    check_initialization_params(attr=return_type, accepted_list=["numpy", "original"])

    mol = Chem.MolFromSmiles(smiles_str)
    fpgen = AllChem.GetMorganGenerator(fpSize=nBits, radius=radius)

    if return_type == "numpy":
        # NOTE: Since it is binary, dtype=np.int8 is possible. BUT it gives error with cuml
        # arr = np.zeros((1,), dtype=np.float32)
        # AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

        return fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)

    elif return_type == "original":
        # return = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return fpgen.GetFingerprint(mol)


def get_ecfp_descriptors(
    smiles_series: pd.Series,
    radius: int = 2,
    nBits: int = 2048,
    inner_return_type="numpy",
) -> Union[np.array, List[ExplicitBitVect]]:
    """
    TODO: Make this faster
    """
    check_initialization_params(
        attr=inner_return_type, accepted_list=["numpy", "original"]
    )

    if inner_return_type == "numpy":
        # NOTE: Since it is binary, dtype=np.int8 is possible. BUT it gives error with cuml
        descriptors = np.zeros((smiles_series.shape[0], nBits), dtype=np.float32)

        for i, smiles in enumerate(smiles_series):
            descriptors[i, :] = get_ecfp_descriptor(
                smiles_str=smiles,
                radius=radius,
                nBits=nBits,
                return_type=inner_return_type,
            )

    elif inner_return_type == "original":
        # descriptors = np.zeros((smiles_series.shape[0], 1), dtype=object)
        smiles_series = smiles_series.to_numpy()

        descriptors = [
            get_ecfp_descriptor(
                smiles_str=smiles_str,
                radius=radius,
                nBits=nBits,
                return_type=inner_return_type,
            )
            for smiles_str in smiles_series
        ]

        # NOTE: Doing this, breaks Tanimoto similarity calculation
        # Because it converts `rdkit.DataStructs.cDataStructs.ExplicitBitVect` to numpy array
        # descriptors = np.array(descriptors)

    return descriptors


#######################################################################################


def get_largest_fragment_from_smiles(s: str):
    """Returns the largest fragment of a SMILES string

    NOTE:
        - Not used right now!
        - From https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/visualization/viz_utils.py
        - Can be used to remove extra fragments in SMILES (typically salts, which are irrelevant to BBB permeability)
    """
    mol = Chem.MolFromSmiles(s)
    if mol:
        clean_mol = LargestFragmentChooser().choose(mol)
        return Chem.MolToSmiles(clean_mol)
    return None


def plot_global_embeddings_with_clusters(  # noqa: PLR0913
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    title: str = "",
    x_lim=None,
    y_lim=None,
):
    """Plots data colored by soft HDBSCAN clusters

    If the cluster of a compund is not specified (cluster < 0), it will be
    plotted gray, otherwise it will be colored by the cluster value.

    NOTE:
        - Not used right now!
        - From https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/visualization/viz_utils.py
    """
    clustered = df[cluster_col].values >= 0

    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        data=df.iloc[~clustered],
        x=x_col,
        y=y_col,
        color=(0.5, 0.5, 0.5),
        s=10,
        alpha=0.1,
    )
    sns.scatterplot(
        data=df.iloc[clustered],
        x=x_col,
        y=y_col,
        hue=cluster_col,
        alpha=0.5,
        palette="nipy_spectral",
        ax=ax,
    )
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    sm = plt.cm.ScalarMappable(cmap="nipy_spectral")
    sm.set_array([])
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="Global Cluster")

    plt.title(title)
    plt.show()


def plot_example_clusters_with_silhouette_samples():
    """Shows side-by-side plot of original and clustered of a mock data"""

    # Generate example data
    X, y = make_blobs(n_samples=80, centers=4, n_features=2, random_state=7)

    # Fit KMeans clustering model
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)

    # Calculate silhouette score
    # silhouette_avg = silhouette_score(X, kmeans.labels_)

    # Create a figure with three subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    # Plot the original data on the left subplot
    axs[0].scatter(X[:, 0], X[:, 1])
    axs[0].set_title("Original Data")
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)

    # Plot the clustered data on the middle subplot
    axs[1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    axs[1].set_title("Clustered Data")
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)

    # Plot a circle around each cluster center
    for i in range(kmeans.n_clusters):
        circle = plt.Circle(
            (kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1]),
            2.6,
            color="red",  # changed color to red
            fill=False,
            linestyle="--",
            linewidth=2.5,
        )
        axs[1].add_artist(circle)

    # Plot the silhouette graph on the right subplot
    sample_silhouette_values = silhouette_samples(X, kmeans.labels_)
    y_lower = 10
    for i in range(kmeans.n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.get_cmap("Spectral")(float(i) / kmeans.n_clusters)
        axs[2].fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        axs[2].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    axs[2].set_title("Silhouette plot")
    # axs[2].set_xlabel("Silhouette coefficient values")
    # axs[2].set_ylabel("Cluster label")

    # Set tick labels to empty
    axs[0].xaxis.set_ticklabels([])
    axs[0].yaxis.set_ticklabels([])
    axs[1].xaxis.set_ticklabels([])
    axs[1].yaxis.set_ticklabels([])
    axs[2].xaxis.set_ticklabels([])
    axs[2].yaxis.set_ticklabels([])

    # Add silhouette score to the title of the silhouette plot
    axs[2].set_title("Silhouette plot")

    plt.savefig("clustered_data.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_choosing_dim_reduction_with_mean(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    base_result = df[df["name"] == "K-MEANS_CHEMBERTA-77M-MTR"]

    ax.plot(
        base_result["n_clusters"],
        base_result["silhouette"],
        color="red",
        label="Baseline",
    )

    colors = ["black", "blue"]
    for i, method in enumerate(["UMAP", "PCA"]):
        processed_result = df[
            (df["name"].str.contains(f"{method}_16"))
            | (df["name"].str.contains(f"{method}_32"))
        ].loc[~df["name"].str.contains("AGGLOMERATIVE")]

        for n in processed_result.name.unique():
            ax.scatter(
                processed_result.loc[processed_result["name"] == n, "n_clusters"],
                processed_result.loc[processed_result["name"] == n, "silhouette"],
                label=n,
                alpha=0.7,
            )

        ax.plot(
            processed_result.groupby(["n_clusters"]).mean().reset_index()["n_clusters"],
            processed_result.groupby(["n_clusters"]).mean().reset_index()["silhouette"],
            color=colors[i],
            label="K-Means Average",
        )

    fig.tight_layout(w_pad=2)
    fig.legend(loc="lower right")
