"""
NOTE:
    - casting to int64 is necessary for cuml metrics
    To fix TypeError: Expected input to be of type in [dtype('int64')] but got int32
"""

import logging
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


logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetric:
    name: str
    function: callable
    need_true_labels: bool


def is_valid_cluster_labels(labels):
    """Check if cluster labels are valid.

    This happens mostly on BUTINA clustering.
    """
    n_samples = len(labels)
    calculated_max_cluster = labels.max()
    MIN_CLUSTER = 2

    if calculated_max_cluster < MIN_CLUSTER or calculated_max_cluster >= n_samples:
        logger.error(
            f"Calculated clusters is {calculated_max_cluster}. Valid values are:"
            f"{MIN_CLUSTER} to {n_samples - 1} (inclusive)."
            "Hence, cluster metrics won't be calculated."
        )
        return False

    return True


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

    TODO:
        - Check maximum data / chunksize ratio with the same accuracy.
            - Up to 5x it is OK right now.
    """
    check_device(device=device)

    if device == "cpu":
        return sklearn_silhouette_score(target, labels, metric="euclidean")

    # NOTE: This changes wrt data and model type.
    chunksize = 30_000

    MAX_TRIES = 3

    while True:
        if MAX_TRIES == 0:
            logger.error(
                "GPU Silhouette score calculation failed with 3 tries."
                "Hence, using sklearn version."
            )
            return sklearn_silhouette_score(target, labels, metric="euclidean")

        try:
            return cuml_silhouette_score(
                target, labels, metric="euclidean", chunksize=chunksize
            )
        except MemoryError:
            logger.error(
                f"When calculating silhouette score with chunksize: {chunksize}, "
                "CUDA out of memory. Trying with smaller chunksize."
            )

            chunksize = chunksize - 5_000
            MAX_TRIES -= 1


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
