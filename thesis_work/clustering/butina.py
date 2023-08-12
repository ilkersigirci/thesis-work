import logging

import matplotlib.pyplot as plt
import pandas as pd
from rdkit.ML.Cluster import Butina

from thesis_work.clustering.utils import get_efcp_distance_matrix

logger = logging.getLogger(__name__)


# TODO: Implement Butina clustering algorithm
def apply_butina(smiles: pd.Series, threshold: float = 0.8):
    distances = get_efcp_distance_matrix(smiles_list=smiles, method="fast")
    clusters = Butina.ClusterData(distances, len(smiles), threshold, isDistData=True)

    return clusters


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
