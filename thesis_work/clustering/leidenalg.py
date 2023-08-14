import igraph as ig
import leidenalg as la
import numpy as np


def apply_leidenalg(distance_matrix):
    """Apply leidenalg clustering on given distance matrix."""
    g = ig.Graph.Weighted_Adjacency(distance_matrix.tolist(), mode="upper")
    partition = la.find_partition(g, la.ModularityVertexPartition)
    cluster_labels = np.array(partition.membership)

    return cluster_labels, None
