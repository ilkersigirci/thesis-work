import logging
import os
import time

import wandb

from thesis_work.clustering.runner import ClusterRunner
from thesis_work.utils.data import (
    load_ataberk,
    load_protein_family_multiple_interacted,  # noqa: F401
    load_related_work,  # noqa: F401
)

logger = logging.getLogger(__name__)


def main():  # noqa: C901, PLR0912, PLR0915
    # NOTE: To disable all wandb logging
    # os.environ["WANDB_MODE"] = "disabled"

    # NOTE: Needed for scalene profiling
    # os.environ["WANDB__EXECUTABLE"] = "/home/ilker/miniconda3/envs/thesis-work/bin/python"

    ############################## GENERIC PARAMS ##############################

    num_threads = None
    random_state = 42
    logged_plot_type = "static"

    device = "cuda"
    # device = "cpu"

    sample_size = None
    # sample_size = 300
    # sample_size = 2_000
    # sample_size = 25_000
    # sample_size = 40_000

    ############################## DATA LOADING ##############################

    # protein_types = [
    #     "gpcr",
    #     "ionchannel",
    #     "kinase",
    #     "nuclearreceptor",
    #     "protease",
    #     "transporter",
    # ]
    # protein_types.sort()
    protein_types = None

    # smiles_df = load_protein_family_multiple_interacted(
    #     protein_types=protein_types,
    #     sample_size=sample_size,
    #     random_state=random_state,
    #     convert_labels=False,
    # )

    # smiles_df = load_related_work(sample_size=sample_size, random_state=random_state)

    subfolder = "chembl27"
    # subfolder = "dude"
    # compound_name = "abl1"
    # compound_name = "renin"
    compound_name = "thb"

    # subfolder = "zinc15"
    # compound_name = None

    smiles_df = load_ataberk(
        subfolder=subfolder,
        compound_name=compound_name,
        return_vectors=False,
        sample_size=sample_size,
        random_state=random_state,
    )

    ############################## OTHER PARAMS ##############################

    wandb_project_name = "ataberk"

    model_with_dims = {
        # "DeepChem/ChemBERTa-77M-MTR": 384,
        # "DeepChem/ChemBERTa-77M-MLM": 384,
        "chemprop": 25,
        "ecfp": 2048,
    }

    dimensionality_reduction_methods = [None, "PCA", "UMAP"]
    n_components_list = [16, 32]

    clustering_method_kwargs_mapping = {
        # "K-MEANS": {
        #     "init_method": "k-means++",
        #     "n_clusters": 3,
        #     "n_init": 1,
        # },
        # "HDBSCAN": {
        #     "min_cluster_size": 5,
        #     "metric": "euclidean",
        #     # "metric": "jaccard",  # NOTE: Doesn't work
        # },
        "AGGLOMERATIVE": {
            "n_clusters": 3,
            "affinity": "euclidean",
            "linkage": "single",
        },
        # "BUTINA": {
        #     # "distance_metric": "euclidean",
        #     "distance_metric": "jaccard",
        #     "threshold": 0.35,
        # },
    }

    if compound_name is not None:
        wandb_project_name += f"-{subfolder}-{compound_name}"
    elif subfolder is not None:
        wandb_project_name += f"-{subfolder}"

    wandb_extra_configs = {}

    if protein_types:
        wandb_extra_configs["proteins"] = protein_types

    ########################## INNER MULTIPLE RUN PARAMS #########################
    # n_clusters = None
    # n_clusters = list(range(2, 100, 3))
    n_clusters = list(range(5, 100, 5))

    # thresholds = None
    # thresholds = [0.5, 0.8]
    thresholds = [0.2, 0.35, 0.5, 0.8]

    # min_samples = None
    min_samples = list(range(5, 100, 5))

    # min_cluster_sizes = None
    min_cluster_sizes = list(range(5, 100, 5))

    input_clustering_param_dict = {
        "n_clusters": n_clusters,
        "threshold": thresholds,
        "min_samples": min_samples,
        "min_cluster_size": min_cluster_sizes,
    }

    ############################## EXPERIMENT LOOP ##############################

    for (
        clustering_method,
        clustering_method_kwargs,
    ) in clustering_method_kwargs_mapping.items():
        for model_name, model_dim in model_with_dims.items():
            for dimensionality_reduction_method in dimensionality_reduction_methods:
                dimensionality_reduction_method_kwargs = None

                if (
                    clustering_method == "BUTINA"
                    and model_name == "ecfp"
                    and dimensionality_reduction_method == "UMAP"
                ):
                    logger.info("Skipping since BUTINA + UMAP + ECFP is not supported")
                    continue

                temp_inner_components_list = (
                    n_components_list
                    if dimensionality_reduction_method is not None
                    else [None]
                )
                for n_components in temp_inner_components_list:
                    if n_components is not None and n_components > model_dim:
                        logger.info(f"Skipping since {n_components} > {model_dim}")
                        continue

                    if dimensionality_reduction_method == "PCA":
                        dimensionality_reduction_method_kwargs = {
                            "n_components": n_components,
                        }
                    elif dimensionality_reduction_method == "UMAP":
                        dimensionality_reduction_method_kwargs = {
                            "n_components": n_components,
                            "n_neighbors": 15,
                            "min_dist": 0.1,
                            "metric": "euclidean",
                            # "metric": "jaccard",
                        }

                    wandb_run_name = f"""
                        {clustering_method}_
                        {model_name if "/" not in model_name else model_name.split("/")[1]}
                    """

                    if dimensionality_reduction_method is not None:
                        wandb_run_name += f"_{dimensionality_reduction_method}"

                    if dimensionality_reduction_method_kwargs is not None:
                        wandb_run_name += (
                            f"_{dimensionality_reduction_method_kwargs['n_components']}"
                        )

                    wandb_extra_configs = {}

                    if protein_types:
                        wandb_extra_configs["proteins"] = protein_types

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

                    start_time = time.time()

                    # cluster_runner.run_clustering()

                    cluster_runner.run_multiple_clustering(
                        input_clustering_param_dict=input_clustering_param_dict
                    )
                    if wandb.run is not None:
                        wandb.finish()

                    end_time = time.time()

                    logger.info(
                        f"****** Time taken: {end_time - start_time} seconds for {wandb_run_name} ******\n"
                    )


if __name__ == "__main__":
    main()
