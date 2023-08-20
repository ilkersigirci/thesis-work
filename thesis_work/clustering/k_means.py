import logging
from typing import Tuple

import numpy as np
from cuml import KMeans as cuKMeans
from sklearn.cluster import KMeans as skKMeans

from thesis_work.utils.utils import check_device

logger = logging.getLogger(__name__)


def apply_k_means(  # noqa: PLR0913
    data: np.array,
    init_method: str = "k-means++",
    n_clusters: int = 2,
    n_init: int = 1,
    random_state: int = 42,
    device: str = "cuda",
) -> Tuple[np.array, np.array]:
    """Fit k-means and return cluster_labels and inertia.

    Args:
        data: Data to be clustered. Vector embeddings of SMILES.
        init_method: Method for initialization, either k-means++ or random.
        n_clusters: Number of clusters.
        random_state: Random state for reproducibility.
        n_init: Number of times the k-means algorithm will be run with different centroid seeds.
        reduced_dimension: Dimensionality of the data after UMAP reduction.
        device: Device to use for clustering, either cpu or cuda.

    Raises:
        ValueError: If device is not either cpu or cuda.
        ValueError: If cuda device is requested but GPU is not available.

    TIMES:
        - 1 to 20 clusters
            CPU: 3.1s
            GPU: 2.2s
        - 1 to 50 clusters
            CPU: 11s
            GPU: 2.1s
    """

    check_device(device=device)

    if device == "cuda" and init_method == "k-means++":
        init_method = "scalable-k-means++"

    K_MEANS = cuKMeans if device == "cuda" else skKMeans

    kmeans = K_MEANS(
        init=init_method,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    kmeans.fit(data)

    return kmeans.labels_, kmeans.inertia_
