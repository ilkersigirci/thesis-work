# ruff: noqa

import logging
import os
import time

from thesis_work.clustering.runner import ClusterRunner
from thesis_work.utils.data import (
    load_protein_family_multiple_interacted,
    load_related_work,
)

logger = logging.getLogger(__name__)


def main():
    # NOTE: To disable all wandb logging
    # os.environ["WANDB_MODE"] = "disabled"

    # NOTE: Needed for scalene profiling
    # os.environ["WANDB__EXECUTABLE"] = "/home/ilker/miniconda3/envs/thesis-work/bin/python"

    wandb_project_name = "related-work"

    num_threads = None
    random_state = 42
    logged_plot_type = "static"
    device = "cuda"

    sample_size = None
    # sample_size = 300
    # sample_size = 10_000
    # sample_size = 40_000

    protein_types = ["gpcr", "kinase", "protease"]
    protein_types.sort()

    # smiles_df = load_protein_family_multiple_interacted(
    #     protein_types=protein_types,
    #     sample_size=sample_size,
    #     random_state=random_state,
    #     convert_labels=False,
    # )

    smiles_df = load_related_work(sample_size=sample_size, random_state=random_state)

    model_name = "DeepChem/ChemBERTa-77M-MTR"
    # model_name = "DeepChem/ChemBERTa-77M-MLM"
    # model_name = "ecfp"
    # model_name = "chemprop"

    # n_components = 25
    n_components = 64

    # dimensionality_reduction_method = None
    # dimensionality_reduction_method_kwargs = None

    dimensionality_reduction_method = "UMAP"
    dimensionality_reduction_method_kwargs = {
        "n_components": n_components,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
    }

    clustering_method = "K-MEANS"
    clustering_method_kwargs = {
        "init_method": "k-means++",
        "n_clusters": 3,
        "n_init": 1,
    }

    # clustering_method = "BUTINA"
    # clustering_method_kwargs = {
    #     # "distance_metric": "tanimoto",
    #     "distance_metric": "euclidean",
    #     "threshold": 0.35,
    # }

    # wandb_run_name = None
    wandb_run_name = f"""
        {clustering_method}_
        {model_name if "/" not in model_name else model_name.split("/")[1]}
    """

    if dimensionality_reduction_method is not None:
        wandb_run_name += f"_{dimensionality_reduction_method}"

    if dimensionality_reduction_method_kwargs is not None:
        wandb_run_name += f"_{dimensionality_reduction_method_kwargs['n_components']}"

    wandb_extra_configs = None
    # wandb_extra_configs = {"proteins": protein_types}

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
        num_threads=num_threads,
        logged_plot_type=logged_plot_type,
    )

    # n_clusters = None
    # n_clusters = list(range(2, 100, 3))
    n_clusters = list(range(5, 100, 5))

    # thresholds = None
    thresholds = [0.2, 0.35, 0.5, 0.8]

    # min_samples = None
    min_samples = [10, 20]

    # min_cluster_sizes = None
    min_cluster_sizes = [5, 10]

    start_time = time.time()

    # cluster_runner.run_clustering()
    cluster_runner.run_multiple_clustering(
        n_clusters=n_clusters,
        thresholds=thresholds,
        min_samples=min_samples,
        min_cluster_sizes=min_cluster_sizes,
    )

    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time}")


if __name__ == "__main__":
    main()
