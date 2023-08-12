import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser


def ignore_warnings(log_level: int = logging.ERROR):
    # Silence transformers warnings
    logging.getLogger("transformers.modeling_utils").setLevel(log_level)
    # from transformers import logging
    # logging.set_verbosity_error()

    # Silence RDKit warnings
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)


def check_device(device: str = "cuda"):
    if device not in ["cpu", "cuda"]:
        raise ValueError("device must be either cpu or cuda")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda device requires GPU")


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
    smiles_str: str, radius: int = 2, nBits: int = 1024, return_type="original"
):
    if not is_valid_smiles(smiles_str):
        raise ValueError("Invalid SMILES string")

    if return_type not in ["original", "numpy", "list"]:
        raise ValueError("Invalid return type")

    mol = Chem.MolFromSmiles(smiles_str)
    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)

    if return_type == "numpy":
        arr = np.zeros((1,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp = arr

        return arr
    elif return_type == "list":
        arr = np.zeros((1,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp = arr.tolist()

    return fp


def get_ecfp_descriptors(
    smiles_series: pd.Series, radius: int = 2, nBits: int = 1024
) -> np.array:
    """
    TODO: Make this faster
    """
    descriptors = np.zeros((smiles_series.shape[0], nBits), dtype=np.int8)

    for i, smiles in enumerate(smiles_series):
        descriptors[i, :] = get_ecfp_descriptor(
            smiles_str=smiles, radius=radius, nBits=nBits, return_type="numpy"
        )

    # NOTE: Don't work
    # vectorized_function = np.vectorize(get_ecfp_descriptor)
    # descriptors = vectorized_function(smiles_series.to_numpy(), radius, nBits)

    return descriptors


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
