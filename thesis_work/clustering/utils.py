import logging
from typing import List

import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances

from thesis_work.utils.initialization import check_initialization_params

# from scipy.spatial import distance_matrix
# from torch import pdist

logger = logging.getLogger(__name__)


def generic_distance_matrix(
    x: np.array,
    metric: str = "euclidean",
    return_upper_tringular: bool = True,
):
    """
    Returns distance matrix for given x and y arrays
    """

    if metric == "tanimoto":
        metric = "jaccard"

        logger.info(
            "Tanimoto metric is same as jaccard metric. Hence, metric changed to jaccard"
        )

    if return_upper_tringular is True:
        return pdist(X=x, metric=metric)

        ## NOTE: Allocates full array, which is unnecessary for only upper triangular matrix
        # And using this somehow increasing running time of butina clustering 5 times,
        # even though matrices are the same.
        # distance_matrix = pairwise_distances(X=x, metric=metric)
        # upper_indices = np.triu_indices(distance_matrix.shape[0], k=1)
        # return distance_matrix[upper_indices]

    else:
        return pairwise_distances(X=x, metric=metric)


def generic_similarity_matrix(
    x: np.array,
    metric: str = "euclidean",
    return_upper_tringular: bool = True,
):
    """
    Returns similarity matrix for given x and y arrays
    """
    return 1 - generic_distance_matrix(
        x=x, metric=metric, return_upper_tringular=return_upper_tringular
    )


def efcp_similarity_matrix(
    ecfps: List[ExplicitBitVect],
    method: str = "fast",
    return_upper_tringular: bool = True,
) -> np.array:
    """
    Returns ECFP similarity matrix for given smiles list

    Args:
        ecfps: List of ECFP fingerprints. Should be created with inner_type=original
            in get_ecfp_descriptors function.

    NOTE:
        - Actually, this function is not neccessary since tanimoto similarity is same as
        jaccard similarity. Hence, we can use generic_similarity_matrix with metric="jaccard".
        - Butina expects, 1d upper triangular matrix. Moreover, this option saves lots of memory.
        - Fast method is ~10x faster than slow method. For 6k compounds:
            - Fast: 3.8 seconds
            - Slow: 39.1 seconds
    """
    check_initialization_params(attr=method, accepted_list=["fast", "slow"])

    n_ecfps = len(ecfps)

    if return_upper_tringular is True:
        similariy_matrix_size = int((n_ecfps * (n_ecfps - 1)) / 2)
        similarity_matrix = np.zeros(similariy_matrix_size)

        last_fill_index = 0
        for i in range(1, n_ecfps):
            last_fill_index += i - 1
            sims = np.array(DataStructs.BulkTanimotoSimilarity(ecfps[i], ecfps[:i]))
            similarity_matrix[last_fill_index : last_fill_index + i] = sims

        return similarity_matrix

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


def efcp_distance_matrix(
    ecfps: List[ExplicitBitVect],
    method: str = "fast",
    return_upper_tringular: bool = True,
) -> np.array:
    """Returns ECFP distance matrix for given smiles list"""

    return 1 - efcp_similarity_matrix(
        ecfps=ecfps,
        method=method,
        return_upper_tringular=return_upper_tringular,
    )
