import pandas as pd
from dagster import EnvVar, RunConfig, job
from dotenv import load_dotenv

from dagster_pipelines.assets.chemberta.data import DataConfig, data_asset
from dagster_pipelines.assets.chemberta.model import (
    MyModelConfig,
    evaluate_model,
    initialize_model,
    train_model,
)
from dagster_pipelines.resources.wandb import WandbResource


@job
def finetune_job() -> None:
    train_df, valid_df, test_df = data_asset()
    initialized_model = initialize_model()
    trained_model = train_model(initialized_model, train_df, valid_df)
    evaluate_model(trained_model, test_df)


if __name__ == "__main__":
    load_dotenv()

    protein_type = "kinase"
    model_type = "DeepChem/ChemBERTa-77M-MLM"

    result = finetune_job.execute_in_process(
        run_config=RunConfig(
            {
                "data_asset": DataConfig(protein_type=protein_type),
                "initialize_model": MyModelConfig(model_type=model_type),
                "train_model": MyModelConfig(
                    model_type=model_type, protein_type=protein_type
                ),
            }
        ),
        resources={"wandb_resource": WandbResource(apikey=EnvVar("WANDB_API_KEY"))},
    )
    assert result.success
    assert isinstance(
        result.output_for_node("data_asset", output_name="train_df_asset"), pd.DataFrame
    )
