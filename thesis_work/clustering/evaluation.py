from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    rand_score,
    silhouette_score,
)


def adjusted_rand_index(target, labels):
    return adjusted_rand_score(target, labels)


def rand_index(target, labels):
    return rand_score(target, labels)


def silhouette_index(target, labels):
    return silhouette_score(target, labels)


def davies_bouldin_index(target, labels):
    return davies_bouldin_score(target, labels)


# TODO: Implement quality partition index
def quality_partition_index():
    pass
