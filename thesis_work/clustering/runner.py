import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import wandb
from thesis_work import LIBRARY_ROOT_PATH
from thesis_work.chemberta.model_descriptors import (
    get_model_descriptors as get_model_descriptors_chemberta,
)
from thesis_work.chemprop.model_descriptors import (
    get_model_descriptors as get_model_descriptors_chemprop,
)
from thesis_work.clustering.butina import apply_butina, calculate_butina_distance_matrix
from thesis_work.clustering.dbscan import apply_hdbscan  # apply_dbscan
from thesis_work.clustering.dimensionality_reduction import (
    apply_pca,
    apply_umap,
    plot_umap,
)
from thesis_work.clustering.evaluation import (
    EvaluationMetric,
    adjusted_rand_index,
    silhouette_index,
)
from thesis_work.clustering.k_means import apply_k_means
from thesis_work.initialization_utils import (
    check_function_init_params,
    check_initialization_params,
    get_function_defaults,
)
from thesis_work.utils import check_device, get_ecfp_descriptors, log_plotly_figure

# from thesis_work.clustering.evaluation import davies_bouldin_index, rand_index

logger = logging.getLogger(__name__)

dimensionality_reduction_mapping = {
    "UMAP": apply_umap,
    "PCA": apply_pca,
}

clustering_algorithm_mapping = {
    "K-MEANS": apply_k_means,
    "BUTINA": apply_butina,
    # "DBSCAN": apply_dbscan, # FIXME: Not working right now.
    "HDBSCAN": apply_hdbscan,
    # "WARD": None,
}

clustering_evaluation_methods = [
    EvaluationMetric(
        name="silhouette", function=silhouette_index, need_true_labels=False
    ),
    EvaluationMetric(
        name="adjusted-rand-index", function=adjusted_rand_index, need_true_labels=True
    ),
    # EvaluationMetric(name="davies-bouldin", function=davies_bouldin_index, need_true_labels=False),
    # EvaluationMetric(name="quality-partition-index", function=quality_partition_index, need_true_labels=True),
]


class ClusterRunner:
    def __init__(  # noqa: PLR0913
        self,
        wandb_project_name: str,
        wandb_run_name: Optional[str] = None,
        wandb_extra_configs: Optional[Dict] = None,
        smiles_df: Optional[pd.DataFrame] = None,
        smiles_df_path: Optional[str] = None,
        model_name: str = "DeepChem/ChemBERTa-77M-MTR",
        random_state: int = 42,
        device: str = "cuda",
        dimensionality_reduction_method: Optional[str] = "UMAP",
        dimensionality_reduction_method_kwargs: Optional[Dict] = None,
        clustering_method: str = "K-MEANS",
        clustering_method_kwargs: Optional[Dict] = None,
        num_threads: Optional[int] = None,
    ):
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.wandb_extra_configs = wandb_extra_configs or {}
        self.smiles_df = smiles_df
        self.smiles_df_path = smiles_df_path
        self.model_name = model_name
        self.random_state = random_state
        self.device = device
        self.dimensionality_reduction_method = (
            dimensionality_reduction_method and dimensionality_reduction_method.upper()
        )
        self.dimensionality_reduction_method_kwargs = (
            dimensionality_reduction_method_kwargs or {}
        )
        self.clustering_method = clustering_method.upper()
        self.clustering_method_kwargs = clustering_method_kwargs or {}
        self.num_threads = num_threads

        self._init_checks()

        wandb.login()

        if self.smiles_df_path is not None:
            self.smiles_df = pd.read_csv(self.smiles_df_path)

        run = wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            dir=LIBRARY_ROOT_PATH,
            config={
                "random_state": self.random_state,
                "device": self.device,
                # "protein_type": None, # TODO: How to make this generic ?
                "dataset_size": self.smiles_df.shape[0],
                "dim_reduction": self.dimensionality_reduction_method,
                **self.dimensionality_reduction_method_kwargs,
                **self.wandb_extra_configs,
            },
        )

        self.vector_embeddings = None
        self.dimensionality_reduction_flag = False
        self.distance_matrix = None

        if self.wandb_run_name is None:
            run.name = run.id
            self.wandb_run_name = run.id

    def __del__(self):
        if wandb.run is not None:
            wandb.finish()

    def _check_global_params(self, in_dict: Dict):
        for global_params in ["random_state", "device"]:
            if in_dict.get(global_params) is not None:
                raise ValueError(
                    f"{global_params} should be provided through the Cluster class"
                )

    def _init_checks(self):
        if self.smiles_df is None and self.smiles_df_path is None:
            raise ValueError("Either smiles_df or smiles_df_path must be provided")

        if self.smiles_df_path is not None:
            self.smiles_df_path = Path(self.smiles_df_path)

            if not self.smiles_df_path.exists():
                raise FileNotFoundError(f"{self.smiles_df_path} does not exist")

        model_names = [
            "DeepChem/ChemBERTa-77M-MTR",
            "DeepChem/ChemBERTa-77M-MLM",
            "chemprop",
            "ecfp",
        ]
        check_initialization_params(attr=self.model_name, accepted_list=model_names)

        check_device(device=self.device)

        check_initialization_params(
            attr=self.dimensionality_reduction_method,
            accepted_list=[None, *list(dimensionality_reduction_mapping.keys())],
        )

        if self.dimensionality_reduction_method is not None:
            self.dimensionality_reduction_func = dimensionality_reduction_mapping[
                self.dimensionality_reduction_method
            ]

            self._check_global_params(
                in_dict=self.dimensionality_reduction_method_kwargs
            )

            check_function_init_params(
                function=self.dimensionality_reduction_func,
                init_params=self.dimensionality_reduction_method_kwargs,
            )
        else:
            self.dimensionality_reduction_func = None

        check_initialization_params(
            attr=self.clustering_method,
            accepted_list=list(clustering_algorithm_mapping.keys()),
        )

        self.clustering_func = clustering_algorithm_mapping[self.clustering_method]

        self._check_global_params(in_dict=self.clustering_method_kwargs)

        check_function_init_params(
            function=self.clustering_func,
            init_params=self.clustering_method_kwargs,
        )

        if (
            self.clustering_method == "BUTINA"
            and self.model_name == "ecfp"
            and self.dimensionality_reduction_func is not None
        ):
            self.dimensionality_reduction_func = None

            message = (
                "Disabling dimensionality reduction, since it is not working "
                "for BUTINA clustering with ecfp model"
            )

            logger.info(message)

    def run_vector_embeddings(self):
        """
        # TODO:
            - Make `radiues` and `nBits` configurable
        """
        if self.vector_embeddings is not None:
            return

        if self.model_name in [
            "DeepChem/ChemBERTa-77M-MTR",
            "DeepChem/ChemBERTa-77M-MLM",
        ]:
            self.vector_embeddings = get_model_descriptors_chemberta(
                smiles_series=self.smiles_df["text"],
                model_name=self.model_name,
                method="simpletransformers",
                device=self.device,
            )
        elif self.model_name == "chemprop":
            self.vector_embeddings = get_model_descriptors_chemprop(
                smiles_series=self.smiles_df["text"]
            )
        elif self.model_name == "ecfp":
            radius = 2
            nBits = 2048

            wandb.config.update({"radius": radius, "nBits": nBits})

            # FIXME: Dimensionality reduction is not working with original
            return_type = "original" if self.clustering_method == "BUTINA" else "numpy"

            self.vector_embeddings = get_ecfp_descriptors(
                smiles_series=self.smiles_df["text"],
                radius=radius,
                nBits=nBits,
                inner_return_type=return_type,
            )

    def run_dimensionality_reduction(self):
        if self.dimensionality_reduction_flag is True:
            return

        if self.dimensionality_reduction_func is not None:
            self.vector_embeddings = self.dimensionality_reduction_func(
                data=self.vector_embeddings,
                **self.dimensionality_reduction_method_kwargs,
            )
            self.dimensionality_reduction_flag = True

    def _run_butina_distance_matrix(self) -> None:
        """
        NOTE
            - Needs ecfp vector embeddings as ExplicitBitVect, not numpy array
        """

        if self.distance_matrix is not None:
            return

        distance_metric = self.clustering_method_kwargs.get("distance_metric", None)

        distance_matrix_kwargs = {
            "data": self.vector_embeddings,
            "model_name": self.model_name,
        }

        if distance_metric is not None:
            distance_matrix_kwargs["distance_metric"] = distance_metric

        self.distance_matrix, nfps = calculate_butina_distance_matrix(
            **distance_matrix_kwargs
        )

        self.clustering_method_kwargs["nfps"] = nfps
        self.clustering_method_kwargs["is_distance_matrix"] = True

        # NOTE: Ecfp vector embeddings don't work with cuml silhoutte since they are list
        self.vector_embeddings = np.array(self.vector_embeddings, dtype=np.float32)

    def log_umap_2D(
        self, data: np.array, labels: pd.Series, log_name: str, legend_title: str
    ) -> None:
        umap_output = apply_umap(
            data=data,
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=self.random_state,
            device=self.device,
        )

        umap_2d_data = pd.DataFrame(
            {"labels": labels, "X": umap_output[:, 0], "Y": umap_output[:, 1]},
            index=labels.index,
        )

        umap_2d_figure = plot_umap(
            data=umap_2d_data,
            plot_title=None,
            legend_title=legend_title,
            method="plotly",
        )
        log_plotly_figure(figure=umap_2d_figure, name=log_name)

    def evaluate_clusters(
        self, cluster_labels: np.array, inertia: Optional[float] = None
    ) -> None:
        """In silhouete score: target: vector_embeddings, labels: cluster_labels
        In adjusted rand index: target: smiles_df["labels"], labels: cluster_labels
        """
        is_true_labels_present = "labels" in self.smiles_df.columns

        for clustering_evaluation_method in clustering_evaluation_methods:
            if not clustering_evaluation_method.need_true_labels:
                target = self.vector_embeddings
            else:
                if not is_true_labels_present:
                    logger.info(
                        f"Skipping calculating {clustering_evaluation_method} since "
                        "true labels are not present in data"
                    )
                    continue

                target = self.smiles_df["labels"]

                # Convert labels to numeric
                if target.dtype == "object":
                    target = pd.factorize(target)[0]

            score = clustering_evaluation_method.function(
                target=target,
                labels=cluster_labels,
                device=self.device,
            )

            logged_data = {clustering_evaluation_method.name: score}

            extra_logged_data_mapping = {
                "K-MEANS": "n_clusters",
                "BUTINA": "threshold",
                "DBSCAN": "min_samples",
                "HDBSCAN": "min_cluster_size",
            }

            if inertia is not None:
                logged_data["inertia"] = inertia

            if self.clustering_method not in extra_logged_data_mapping:
                raise ValueError(
                    f"Extra logged data mapping not present for {self.clustering_method}"
                )

            # logged_data.update(extra_logged_data_mapping[self.clustering_method])
            added_key = extra_logged_data_mapping[self.clustering_method]
            added_value = self.clustering_method_kwargs.get(added_key, None)

            # NOTE: Get default values from function
            if added_value is None:
                added_value = get_function_defaults(function=self.clustering_func).get(
                    added_key
                )

            logged_data[added_key] = added_value

            wandb.log(
                data=logged_data,
                # step=self.clustering_method_kwargs["n_clusters"],
            )

    def _run_clustering(self):
        self.run_vector_embeddings()
        self.run_dimensionality_reduction()

        # FIXME: If labels not present, this won't work
        # FIXME: legend_title should be different for active-inactive datasets
        self.log_umap_2D(
            data=self.vector_embeddings,
            labels=self.smiles_df["labels"],
            log_name="Original Labels",
            legend_title="Protein Family",
        )

        clustering_kwargs = self.clustering_method_kwargs
        clustering_kwargs["data"] = self.vector_embeddings

        # Don't calculate distance matrix each time
        if self.clustering_method == "BUTINA":
            self._run_butina_distance_matrix()
            clustering_kwargs["data"] = self.distance_matrix
            clustering_kwargs["model_name"] = self.model_name

        cluster_labels, inertia = self.clustering_func(**clustering_kwargs)

        self.evaluate_clusters(cluster_labels=cluster_labels, inertia=inertia)

        # Log with cluster labels
        cluster_labels = pd.Series(cluster_labels, name="labels")
        self.log_umap_2D(
            data=self.vector_embeddings,
            labels=cluster_labels,
            log_name="Cluster Labels",
            legend_title="Protein Family",
        )

    def run_clustering(self):
        if self.num_threads is not None:
            with threadpool_limits(limits=self.num_threads, user_api="blas"):
                self._run_clustering()
        else:
            self._run_clustering()

    def run_multiple_clustering(  # noqa: C901
        self,
        n_clusters: Optional[List[int]] = None,
        thresholds: Optional[List[float]] = None,
        min_samples: Optional[List[int]] = None,
        min_cluster_sizes: Optional[List[int]] = None,
    ):
        if self.clustering_method == "K-MEANS":
            if n_clusters is None:
                n_clusters = [2, 3, 5, 10, 20, 50, 100]

            for n_cluster in n_clusters:
                # NOTE: All clustering functions should implement n_clusters as a parameter
                self.clustering_method_kwargs["n_clusters"] = n_cluster

                self.run_clustering()

        elif self.clustering_method == "BUTINA":
            if thresholds is None:
                thresholds = [0.2, 0.35, 0.5, 0.8]

            for threshold in thresholds:
                self.clustering_method_kwargs["threshold"] = threshold

                self.run_clustering()

        elif self.clustering_method == "DBSCAN":
            if min_samples is None:
                min_samples = [2, 3, 5, 10, 20, 50, 100]

            for min_sample in min_samples:
                self.clustering_method_kwargs["min_samples"] = min_sample

                self.run_clustering()

        elif self.clustering_method == "HDBSCAN":
            if min_cluster_sizes is None:
                min_cluster_size = [2, 3, 5, 10, 20, 50, 100]

            for min_cluster_size in min_cluster_sizes:
                self.clustering_method_kwargs["min_cluster_size"] = min_cluster_size

                self.run_clustering()


if __name__ == "__main__":
    pass
