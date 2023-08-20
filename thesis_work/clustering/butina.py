"""
NOTE:
    - Generating distance matrix needas lots of memory.
    For example, array of size:
        - (100_000, 100) needs 37 GB memory.
        - (1_000_000, 25) needs 3.64 TB memory.
    Hence, we need to consider this limitation when clustering big datasets.
"""


import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.ML.Cluster import Butina

from thesis_work.clustering.utils import generic_distance_matrix

logger = logging.getLogger(__name__)


def calculate_butina_distance_matrix(
    data: np.array, model_name: str, distance_metric: str
):
    if model_name == "ecfp" and distance_metric not in ["tanimoto", "jaccard"]:
        logger.info("For ecfp fingerprints, jaccard(tanimoto) metric is recommended.")

    nfps = data.shape[0]
    distances = generic_distance_matrix(
        x=data, metric=distance_metric, return_upper_tringular=True
    )

    return distances, nfps


def apply_butina(  # noqa: PLR0913
    data: np.array,
    model_name: str,
    distance_metric: str = "euclidean",
    threshold: float = 0.35,
    nfps: Optional[int] = None,
    is_distance_matrix: bool = False,
    random_state: int = 42,
    device: str = "cuda",
) -> Tuple[np.array, None]:
    """Apply butina clustering on given smiles list.

    Args:
        data: Ecfp vector embeddings or distance matrix
        model_name: model that used to calculate distance matrix.
        distance_metric: Distance metric to calculate distance matrix.
        threshold: Threshold for butina clustering.
        nfps: Length of original data, not the distance matrix.
        is_distance_matrix: If True, data is already a distance matrix.

    NOTE:
        - For ecfp, rdkit tanimoto distance matrix calculation isn't used. Instead,
        sklearn pdisk with jaccard distance is used, since they are basically same.
    """
    if is_distance_matrix is True:
        if nfps is None:
            raise ValueError("If is_distance_matrix is True, nfps must be given.")

        distances = data

    else:
        distances, nfps = calculate_butina_distance_matrix(
            data=data, model_name=model_name, distance_metric=distance_metric
        )

    if device == "cuda":
        logger.info("BUTINA doesn't support GPU, hence, it will be run on CPU.")

    # TODO: Add distFunc=distance_metric as function
    clusters = Butina.ClusterData(distances, nfps, threshold, isDistData=True)
    cluster_labels = np.zeros(nfps, dtype=np.uint32)

    for idx, cluster in enumerate(clusters, 1):
        for member in cluster:
            cluster_labels[member] = idx

    return cluster_labels, None


def butina_report(clusters):
    clusters = sorted(clusters, key=len, reverse=True)

    # cluster_bins = np.bincount(clusters)

    ## Give a short report about the numbers of clusters and their sizes
    # num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
    # num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
    # num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
    # num_clust_g100 = sum(1 for c in clusters if len(c) > 100)

    # print("total # clusters: ", len(clusters))
    # print("# clusters with only 1 compound: ", num_clust_g1)
    # print("# clusters with >5 compounds: ", num_clust_g5)
    # print("# clusters with >25 compounds: ", num_clust_g25)
    # print("# clusters with >100 compounds: ", num_clust_g100)

    # Plot the size of the clusters
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_xlabel("Cluster index")
    ax.set_ylabel("Number of molecules")
    ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], lw=5)


def custom_butina_only_ecfp(smiles: pd.Series, threshold: float = 0.35):
    """
    From: https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/clustering/taylor_butina_clustering.ipynb
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors as rdmd

    # mol_list = smiles.apply(Chem.MolFromSmiles).tolist()
    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles]

    fp_list = [rdmd.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mol_list]
    dists = []
    nfps = len(fp_list)

    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])

    mol_clusters = Butina.ClusterData(dists, nfps, threshold, isDistData=True)
    cluster_labels = [0] * nfps

    for idx, cluster in enumerate(mol_clusters, 1):
        for member in cluster:
            cluster_labels[member] = idx

    return cluster_labels
