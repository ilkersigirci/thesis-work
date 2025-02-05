"""
NOTE:
    - casting to int64 is necessary for cuml metrics
    To fix TypeError: Expected input to be of type in [dtype('int64')] but got int32
"""

from dataclasses import dataclass

from cuml.metrics.cluster import (
    adjusted_rand_score as cuml_adjusted_rand_score,
    completeness_score as cuml_completeness_score,
    homogeneity_score as cuml_homogeneity_score,
    mutual_info_score as cuml_mutual_info_score,
    silhouette_score as cuml_silhouette_score,
)
from sklearn.metrics import (
    adjusted_rand_score as sklearn_adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score as sklearn_completeness_score,
    davies_bouldin_score,
    homogeneity_score as sklearn_homogeneity_score,
    mutual_info_score as sklearn_mutual_info_score,
    silhouette_score as sklearn_silhouette_score,
)

from thesis_work.utils.utils import check_device

# from sklearn.metrics import rand_score


@dataclass
class EvaluationMetric:
    name: str
    function: callable
    need_true_labels: bool


def adjusted_rand_index(target, labels, device="cuda"):
    check_device(device=device)

    adjusted_rand_score = (
        cuml_adjusted_rand_score if device == "cuda" else sklearn_adjusted_rand_score
    )
    return adjusted_rand_score(target, labels)


def calinski_harabasz_index(target, labels, device="cuda"):
    return calinski_harabasz_score(target, labels)


def completeness_index(target, labels, device="cuda"):
    check_device(device=device)

    target = target.astype("int64")
    labels = labels.astype("int64")

    completeness_score = (
        cuml_completeness_score if device == "cuda" else sklearn_completeness_score
    )
    return completeness_score(target, labels)


def davies_bouldin_index(target, labels, device="cuda"):
    return davies_bouldin_score(target, labels)


def homogeneity_index(target, labels, device="cuda"):
    check_device(device=device)

    target = target.astype("int64")
    labels = labels.astype("int64")

    homogeneity_score = (
        cuml_homogeneity_score if device == "cuda" else sklearn_homogeneity_score
    )
    return homogeneity_score(target, labels)


def mutual_info_index(target, labels, device="cuda"):
    check_device(device=device)

    target = target.astype("int64")
    labels = labels.astype("int64")

    mutual_info_score = (
        cuml_mutual_info_score if device == "cuda" else sklearn_mutual_info_score
    )
    return mutual_info_score(target, labels)


# def rand_index(target, labels, device="cuda"):
#     return rand_score(target, labels)


def silhouette_index(target, labels, device="cuda"):
    """
    NOTE: cuml version needs quadratic memory. To fix it chunksize should be set.
    Default is 40_000 but it exceeds memory limit of my GPU, since it has 8 GB VRAM.
    Moreover, using chunksize doesn't descrease accuracy: https://github.com/rapidsai/cuml/pull/3362
    - sample_size=chunksize is not necessary for sklearn version
    """
    check_device(device=device)

    # Maximum working value for my GPU
    chunksize = 32_000

    if device == "cuda":
        return cuml_silhouette_score(
            target, labels, metric="euclidean", chunksize=chunksize
        )
    elif device == "cpu":
        return sklearn_silhouette_score(target, labels, metric="euclidean")


# TODO: Implement quality partition index
# def quality_partition_index(target, labels, device="cuda"):
#     pass


CLUSTERING_EVALUATION_METRICS = [
    EvaluationMetric(
        name="adjusted-rand-index",
        function=adjusted_rand_index,
        need_true_labels=True,
    ),
    EvaluationMetric(
        name="calinski-harabasz-index",
        function=calinski_harabasz_index,
        need_true_labels=False,
    ),
    EvaluationMetric(
        name="completeness-index",
        function=completeness_index,
        need_true_labels=True,
    ),
    EvaluationMetric(
        name="davies-bouldin",
        function=davies_bouldin_index,
        need_true_labels=False,
    ),
    EvaluationMetric(
        name="homogeneity-index",
        function=homogeneity_index,
        need_true_labels=True,
    ),
    EvaluationMetric(
        name="mutual-info-index",
        function=mutual_info_index,
        need_true_labels=True,
    ),
    EvaluationMetric(
        name="silhouette",
        function=silhouette_index,
        need_true_labels=False,
    ),
    # EvaluationMetric(
    #     name="quality-partition-index",
    #     function=quality_partition_index,
    #     need_true_labels=True,
    # ),
]
