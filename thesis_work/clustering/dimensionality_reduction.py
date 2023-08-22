import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import wandb
from cuml import PCA as pca_gpu, UMAP as umap_gpu
from sklearn.decomposition import PCA as pca_cpu
from umap import UMAP as umap_cpu

from thesis_work.utils.initialization import check_initialization_params
from thesis_work.utils.utils import check_device

logger = logging.getLogger(__name__)


def apply_pca(  # noqa: PLR0913
    data: np.array,
    n_components: int,
    svd_solver="full",
    tol: float = 1e-7,
    random_state: int = 42,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Apply PCA and return reduced dimensionality data.

    Args:
        data: Data to be reduced.
        n_components: Dimensionality of the data after reduction.

    TODO:
        - Test explainability when choosing n_components. Cumulative variance:
        `np.sum(pca.explained_variance_ratio_)))`
    """
    check_device(device=device)

    if n_components > data.shape[1]:
        raise ValueError("n_components must be less than or equal to data.shape[1]")

    if n_components == data.shape[1]:
        logger.warning(
            "n_components is equal to data.shape[1]. Returning original data."
        )
        return data

    PCA = pca_gpu if device == "cuda" else pca_cpu

    pca_model = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        tol=tol,
        random_state=random_state,
    )

    transformed_data = pca_model.fit_transform(data)
    wandb.log({"pca_explained_variance": np.sum(pca_model.explained_variance_ratio_)})

    return transformed_data


def apply_umap(  # noqa: PLR0913
    data: np.array,
    n_components: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Apply UMAP and return reduced dimensionality data.

    Args:
        data: Data to be reduced.
        n_components: Dimensionality of the data after reduction.
        n_neighbors: Number of neighbors to use for UMAP reduction.

    CPU  Only Params: low_memory=False

    TIMES:
        - TODO: Add times for CPU and GPU

    TODO:
        - Change default: n_epochs=None, learning_rate=1.0
            - n_epochs (default): 200 for large datasets, 500 for small datasets
        - For cpu UMAP, add parametric UMAP
        - Implement parametric UMAP with Autoencoder
            https://github.com/lmcinnes/umap/blob/master/notebooks/Parametric_UMAP/04.0-parametric-umap-mnist-embedding-convnet-with-autoencoder-loss.ipynb
        -

    NOTE:
        - cuml version has only supported "jaccard" distance metric for sparse inputs.
    """
    check_device(device=device)

    if n_components > data.shape[1]:
        raise ValueError("n_components must be less than or equal to data.shape[1]")

    if n_components == data.shape[1]:
        logger.warning(
            "n_components is equal to data.shape[1]. Returning original data."
        )
        return data

    UMAP = umap_gpu if device == "cuda" else umap_cpu

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    return umap_model.fit_transform(data)


def plot_umap(
    data: pd.DataFrame,
    plot_title: Optional[str] = None,
    legend_title: str = "Protein Family",
    method="plotly",
):
    check_initialization_params(
        attr=method, accepted_list=["plotly", "matplotlib", "seaborn"]
    )
    if set(data.columns) != {"X", "Y", "labels"}:
        raise ValueError("data columns must be ['X', 'Y', 'labels']")

    if method == "plotly":
        fig = px.scatter(data_frame=data, x="X", y="Y", color="labels", opacity=0.8)
        fig.update_layout(legend_title=legend_title)

    elif method == "matplotlib":
        fig = plt.scatter(
            data[:, 0], data[:, 1], c=data["labels"], s=0.1, cmap="Spectral"
        )

        if plot_title is not None:
            plt.title(plot_title)

        plt.legend(title=legend_title, loc="upper right", labels=data["labels"])

    elif method == "seaborn":
        unique_labels = data["labels"].unique().shape[0]
        palette = sns.color_palette("bright", unique_labels)

        # Seaborn way
        fig = plt.figure(figsize=(8, 8))
        _ = sns.scatterplot(
            data=data,
            x="X",
            y="Y",
            hue="labels",
            alpha=0.5,
            palette=palette,
        )
        if plot_title is not None:
            plt.title(plot_title)

        plt.legend(title=legend_title, loc="upper right", labels=data["labels"])
        # plt.show()

    return fig
