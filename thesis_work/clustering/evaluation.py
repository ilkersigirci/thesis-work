from dataclasses import dataclass

from cuml.metrics.cluster import (
    adjusted_rand_score as cuml_adjusted_rand_score,
    silhouette_score as cuml_silhouette_score,
)
from sklearn.metrics import (
    adjusted_rand_score as sklearn_adjusted_rand_score,
    davies_bouldin_score,
    rand_score,
    silhouette_score as sklearn_silhouette_score,
)

from thesis_work.utils import check_device


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


def rand_index(target, labels, device="cuda"):
    return rand_score(target, labels)


def silhouette_index(target, labels, device="cuda"):
    check_device(device=device)

    silhouette_score = (
        cuml_silhouette_score if device == "cuda" else sklearn_silhouette_score
    )

    return silhouette_score(target, labels)


def davies_bouldin_index(target, labels, device="cuda"):
    return davies_bouldin_score(target, labels)


# TODO: Implement quality partition index
def quality_partition_index(target, labels, device="cuda"):
    pass
