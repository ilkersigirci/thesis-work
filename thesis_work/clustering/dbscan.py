from typing import Tuple

import numpy as np
import torch
from cuml import DBSCAN as cuDBSCAN, HDBSCAN as cuHDBSCAN
from sklearn.cluster import DBSCAN as skDBSCAN, HDBSCAN as skHDBSCAN

from thesis_work.utils.utils import check_device


def apply_dbscan(
    data: np.array,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    device: str = "cuda",
) -> Tuple[np.array, None]:
    """Apply dbscan.

    Args:
        data: _description_.
        eps: _description_.
        min_samples: _description_.
        metric: _description_.
        device: _description_.

    Returns:
        Cluster labels.

    FIXME
        - Doesn't work with silhoutte score.
    """

    check_device(device=device)

    DBSCAN = cuDBSCAN if torch.cuda.is_available() else skDBSCAN

    # CPU only params: algorithm: str = "auto"
    clustering_model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )

    clustering_model.fit(data)

    return clustering_model.labels_, None


def apply_hdbscan(  # noqa: PLR0913
    data: np.array,
    min_cluster_size: int = 5,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    device: str = "cuda",
) -> Tuple[np.array, None]:
    check_device(device=device)

    HDBSCAN = cuHDBSCAN if torch.cuda.is_available() else skHDBSCAN

    # CPU only params: algorithm: str = "auto"
    clustering_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )

    clustering_model.fit(data)

    return clustering_model.labels_, None
