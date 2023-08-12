import logging
from typing import List

import numpy as np
from rdkit import DataStructs

from thesis_work.utils import get_ecfp_descriptor

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
