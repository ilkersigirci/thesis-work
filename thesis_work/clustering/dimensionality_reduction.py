import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from cuml import PCA as pca_gpu, UMAP as umap_gpu
from sklearn.decomposition import PCA as pca_cpu
from umap import UMAP as umap_cpu

import wandb
from thesis_work.utils import check_device

logger = logging.getLogger(__name__)


def apply_pca(  # noqa: PLR0913
    data: np.array,
    n_components: int = 2,
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

    PCA = pca_gpu if torch.cuda.is_available() else pca_cpu

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
