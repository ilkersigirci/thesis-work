import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from cuml import UMAP as umap_gpu
from rdkit import DataStructs
from umap import UMAP as umap_cpu

from thesis_work.utils import check_device, get_ecfp_descriptor

logger = logging.getLogger(__name__)


def get_efcp_similarity_matrix(
    smiles_list: List[str], method: str = "fast"
) -> np.array:
    """
    Returns ECFP similarity matrix for given smiles list

    NOTE:
        - Fast method is ~10x faster than slow method. For 6k compounds:
            - Fast: 3.8 seconds
            - Slow: 39.1 seconds
    """

    if method not in ["fast", "slow"]:
        raise ValueError("method must be either fast or slow")

    # Generate ECFP descriptors
    ecfps = [
        get_ecfp_descriptor(smiles_str=smiles_str, nBits=2048)
        for smiles_str in smiles_list
    ]

    n_ecfps = len(ecfps)

    if method == "fast":
        similarity_matrix = np.zeros((n_ecfps, n_ecfps))

        for i in range(1, n_ecfps):
            similarity = DataStructs.BulkTanimotoSimilarity(ecfps[i], ecfps[:i])
            similarity_matrix[i, :i] = similarity
            similarity_matrix[:i, i] = similarity
            np.fill_diagonal(similarity_matrix, 1)

    elif method == "slow":
        similarity_matrix = []
        for i in range(n_ecfps):
            temp_similarities = []
            for j in range(n_ecfps):
                similarity = DataStructs.TanimotoSimilarity(ecfps[i], ecfps[j])
                temp_similarities.append(similarity)
            similarity_matrix.append(temp_similarities)

        similarity_matrix = np.array(similarity_matrix)

    return similarity_matrix


def get_efcp_distance_matrix(smiles_list: List[str], method: str = "fast"):
    similarity_matrix = get_efcp_similarity_matrix(
        smiles_list=smiles_list, method=method
    )

    distance_matrix = 1 - similarity_matrix

    return distance_matrix


def apply_umap(  # noqa: PLR0913
    data: np.array,
    n_components: int = 2,
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
    """
    check_device(device=device)

    if n_components > data.shape[1]:
        raise ValueError("n_components must be less than or equal to data.shape[1]")

    UMAP = umap_gpu if torch.cuda.is_available() else umap_cpu

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        # metric="jaccard",
        random_state=random_state,
    )

    return umap_model.fit_transform(data)


def plot_umap(
    data: pd.DataFrame,
    labels: np.array,
    legend_title: str,
    plot_title: Optional[str] = None,
) -> None:
    """Plot UMAP."""
    palette = sns.color_palette("bright", 3)

    plt.figure(figsize=(8, 8))
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

    plt.legend(title=legend_title, loc="upper right", labels=labels)
    plt.show()
