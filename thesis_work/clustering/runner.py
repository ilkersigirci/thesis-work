import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
from thesis_work.clustering.dimensionality_reduction import apply_pca, apply_umap
from thesis_work.clustering.evaluation import (
    adjusted_rand_index,
    silhouette_index,
)
from thesis_work.clustering.k_means import apply_k_means
from thesis_work.initialization_utils import (
    check_function_init_params,
    check_initialization_params,
)
from thesis_work.utils import check_device, get_ecfp_descriptors

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

clustering_evaluation_method_mapping = {
    "silhouette": silhouette_index,
    "adjusted-rand-index": adjusted_rand_index,
    # "davies-bouldin": None,
    # "quality-partition-index": None,
}


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
        clustering_evaluation_method: str = "silhouette",
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
        self.clustering_evaluation_method = clustering_evaluation_method

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

        check_initialization_params(
            attr=self.clustering_evaluation_method,
            accepted_list=list(clustering_evaluation_method_mapping.keys()),
        )

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

    def run_clustering(self):
        self.run_vector_embeddings()
        self.run_dimensionality_reduction()

        clustering_kwargs = self.clustering_method_kwargs
        clustering_kwargs["data"] = self.vector_embeddings

        # Don't calculate distance matrix each time
        if self.clustering_method == "BUTINA":
            self._run_butina_distance_matrix()
            clustering_kwargs["data"] = self.distance_matrix
            clustering_kwargs["model_name"] = self.model_name

            # NOTE: Ecfp vector embeddings don't work with cuml silhoutte since they are list
            self.vector_embeddings = np.array(self.vector_embeddings, dtype=np.float32)

        cluster_labels, inertia = self.clustering_func(**clustering_kwargs)
        cluster_evaluation_func = clustering_evaluation_method_mapping[
            self.clustering_evaluation_method
        ]
        score = cluster_evaluation_func(
            target=np.array(self.vector_embeddings, dtype=np.float32),
            labels=cluster_labels,
            device=self.device,
        )

        if self.clustering_method == "K-MEANS":
            wandb.log(
                data={
                    f"{self.clustering_evaluation_method}_score": score,
                    "inertia": inertia,
                    "n_clusters": self.clustering_method_kwargs["n_clusters"],
                },
                # step=self.clustering_method_kwargs["n_clusters"],
            )

        elif self.clustering_method == "BUTINA":
            wandb.log(
                data={
                    f"{self.clustering_evaluation_method}_score": score,
                    "threshold": self.clustering_method_kwargs["threshold"],
                },
                # step=self.clustering_method_kwargs["threshold"],
            )

        elif self.clustering_method == "DBSCAN":
            wandb.log(
                data={
                    f"{self.clustering_evaluation_method}_score": score,
                    "min_samples": self.clustering_method_kwargs["min_samples"],
                },
                # step=self.clustering_method_kwargs["min_samples"],
            )

        elif self.clustering_method == "HDBSCAN":
            wandb.log(
                data={
                    f"{self.clustering_evaluation_method}_score": score,
                    "min_cluster_size": self.clustering_method_kwargs[
                        "min_cluster_size"
                    ],
                },
                # step=self.clustering_method_kwargs["min_cluster_size"],
            )

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
    import os
    import time

    import pandas as pd

    from thesis_work.data import load_mixed_interacted_compounds

    ## To disable all wandb logging
    os.environ["WANDB_MODE"] = "disabled"

    wandb_project_name = "clustering-class-test"

    random_state = 42
    device = "cuda"

    each_sample_size = 1000
    protein_types = ["gpcr", "kinase", "protease"]
    protein_types.sort()
    protein_labels = list(range(len(protein_types)))

    smiles_df = load_mixed_interacted_compounds(
        protein_types=protein_types,
        each_sample_size=each_sample_size,
        random_state=random_state,
        convert_category=True,
    )

    model_name = "DeepChem/ChemBERTa-77M-MTR"
    # model_name = "DeepChem/ChemBERTa-77M-MLM"
    # model_name = "ecfp"
    # model_name = "chemprop"

    n_components = 25

    # dimensionality_reduction_method = None
    # dimensionality_reduction_method_kwargs = None

    dimensionality_reduction_method = "UMAP"
    dimensionality_reduction_method_kwargs = {
        "n_components": n_components,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
    }

    # clustering_method = "K-MEANS"
    # clustering_method_kwargs = {
    #     "init_method": "k-means++",
    #     "n_clusters": 3,
    #     "n_init": 1,
    # }

    clustering_method = "BUTINA"
    clustering_method_kwargs = {
        # "distance_metric": "tanimoto",
        "distance_metric": "euclidean",
        "threshold": 0.35,
    }

    # wandb_run_name = None
    wandb_run_name = f"""
        {clustering_method}_
        {model_name if "/" not in model_name else model_name.split("/")[1]}
    """

    if dimensionality_reduction_method is not None:
        wandb_run_name += f"_{dimensionality_reduction_method}"

    if dimensionality_reduction_method_kwargs is not None:
        wandb_run_name += f"_{dimensionality_reduction_method_kwargs['n_components']}"

    # wandb_extra_configs = None
    wandb_extra_configs = {"proteins": protein_types}

    cluster_runner = ClusterRunner(
        wandb_project_name=wandb_project_name,
        wandb_run_name=wandb_run_name,
        wandb_extra_configs=wandb_extra_configs,
        smiles_df=smiles_df,
        # smiles_df_path = None,
        model_name=model_name,
        random_state=random_state,
        device=device,
        dimensionality_reduction_method=dimensionality_reduction_method,
        dimensionality_reduction_method_kwargs=dimensionality_reduction_method_kwargs,
        clustering_method=clustering_method,
        clustering_method_kwargs=clustering_method_kwargs,
        clustering_evaluation_method="silhouette",
    )

    # n_clusters = None
    n_clusters = list(range(2, 100, 3))

    # thresholds = None
    thresholds = [0.2, 0.35, 0.5, 0.8]

    # min_samples = None
    min_samples = [10, 20]

    # min_cluster_sizes = None
    min_cluster_sizes = [5, 10]

    start_time = time.time()

    cluster_runner.run_clustering()
    # cluster_runner.run_multiple_clustering(
    #     n_clusters=n_clusters,
    #     thresholds=thresholds,
    #     min_cluster_sizes=min_cluster_sizes,
    # )

    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time}")
