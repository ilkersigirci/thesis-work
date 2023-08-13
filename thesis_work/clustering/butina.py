import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.ML.Cluster import Butina

from thesis_work.clustering.utils import (
    efcp_distance_matrix,
    generic_distance_matrix,
)
from thesis_work.initialization_utils import check_initialization_params

logger = logging.getLogger(__name__)


def apply_butina(
    data: np.array,  # Ecfp vector embeddings
    method: str = "generic",
    distance_metric: str = "euclidian",
    threshold: float = 0.35,
) -> Tuple[np.array, None]:
    """Apply butina clustering on given smiles list."""
    check_initialization_params(attr=method, accepted_list=["generic", "ecfp"])

    if method == "ecfp" and distance_metric != "tanimoto":
        distance_metric = "tanimoto"
        message = (
            "For ecfp distance matrix calculation, only tanimoto is supported."
            "Hence, distance_metric changec to tanimoto"
        )
        logger.info(message)

    if method == "generic":
        nfps = data.shape[0]
        distances = generic_distance_matrix(x=data, metric=distance_metric)
    elif method == "ecfp":
        nfps = len(data)
        distances = efcp_distance_matrix(
            ecfps=data, method="fast", return_upper_tringular=True
        )

    # TODO: Add distFunc=distance_metric as function
    clusters = Butina.ClusterData(distances, nfps, threshold, isDistData=True)
    cluster_labels = np.zeros(nfps, dtype=np.uint32)

    for idx, cluster in enumerate(clusters, 1):
        for member in cluster:
            cluster_labels[member] = idx

    return cluster_labels, None


def butina_report(clusters):
    clusters = sorted(clusters, key=len, reverse=True)

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
