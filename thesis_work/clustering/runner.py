from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import wandb
from thesis_work import LIBRARY_ROOT_PATH
from thesis_work.chemberta.model_descriptors import (
    get_model_descriptors as get_model_descriptors_chemberta,
)
from thesis_work.chemprop.model_descriptors import (
    get_model_descriptors as get_model_descriptors_chemprop,
)
from thesis_work.clustering.butina import apply_butina
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

dimensionality_reduction_mapping = {
    "UMAP": apply_umap,
    "PCA": apply_pca,
}

clustering_algorithm_mapping = {
    "K-MEANS": apply_k_means,
    "BUTINA": apply_butina,
    # "WARD": None,
}

clustering_evaluation_method_mapping = {
    "silhouette": silhouette_index,
    "adjusted-rand-index": adjusted_rand_index,
    # "davies-bouldin": None,
    "quality-partition-index": None,
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

            # NOTE: Dimensionality reduction not working with original
            inner_return_type = (
                "original" if self.clustering_method == "BUTINA" else "numpy"
            )

            self.vector_embeddings = get_ecfp_descriptors(
                smiles_series=self.smiles_df["text"],
                radius=radius,
                nBits=nBits,
                inner_return_type=inner_return_type,
            )

    def run_dimensionality_reduction(self):
        if self.dimensionality_reduction_flag is True:
            return

        if self.dimensionality_reduction_func is not None:
            self.vector_embeddings = self.dimensionality_reduction_func(
                self.vector_embeddings
            )
            self.dimensionality_reduction_flag = True

    def run_clustering(self):
        self.run_vector_embeddings()
        self.run_dimensionality_reduction()

        cluster_labels, inertia = self.clustering_func(
            data=self.vector_embeddings, **self.clustering_method_kwargs
        )
        cluster_evaluation_func = clustering_evaluation_method_mapping[
            self.clustering_evaluation_method
        ]

        score = cluster_evaluation_func(self.vector_embeddings, cluster_labels)

        if self.clustering_method == "K-MEANS":
            wandb.log(
                data={
                    f"{self.clustering_evaluation_method}_score": score,
                    "inertia": inertia,
                    "n_clusters": self.clustering_method_kwargs["n_clusters"],
                },
                # step=self.clustering_method_kwargs["n_clusters"],
            )

        if self.clustering_method == "BUTINA":
            wandb.log(
                data={
                    f"{self.clustering_evaluation_method}_score": score,
                    "threshold": self.clustering_method_kwargs["threshold"],
                },
                # step=self.clustering_method_kwargs["threshold"],
            )

    def run_multiple_clustering(
        self,
        n_clusters: Optional[List[int]] = None,
        thresholds: Optional[List[float]] = None,
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


if __name__ == "__main__":
    pass
