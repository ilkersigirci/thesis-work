import logging
from typing import Tuple

import numpy as np
import torch
from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering as skAgglomerativeClustering

from thesis_work.utils.utils import check_device

logger = logging.getLogger(__name__)


def apply_agglomerative(  # noqa: PLR0913
    data: np.array,
    n_clusters: int = 2,
    affinity: str = "euclidean",
    linkage: str = "single",
    connectivity="knn",
    device: str = "cuda",
) -> Tuple[np.array, None]:
    """
    NOTE:
        - cuml version doesn't support linkage="ward"
    """
    check_device(device=device)

    AGGLOMEREATIVE = (
        cuAgglomerativeClustering
        if torch.cuda.is_available()
        else skAgglomerativeClustering
    )

    agglomerative = AGGLOMEREATIVE(
        n_clusters=n_clusters,
        affinity=affinity,
        linkage=linkage,
        connectivity=connectivity,
    )
    agglomerative.fit(data)

    return agglomerative.labels_, None
