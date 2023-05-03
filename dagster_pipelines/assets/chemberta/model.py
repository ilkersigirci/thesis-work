# FIXME: Not working in dagster: https://github.com/dagster-io/dagster/issues/8540
# from __future__ import annotations

import pandas as pd
from dagster import (
    Config,
    EnvVar,
    OpExecutionContext,
    RunConfig,
    asset,
    materialize_to_memory,
)
from dotenv import load_dotenv
from pydantic import Field
from simpletransformers.classification import ClassificationModel

from dagster_pipelines.resources.wandb import WandbResource
from thesis_work.chemberta.utils import (
    evaluate_model as evaluate_model_util,
    get_model,
    train_model as train_model_util,
)


class MyModelConfig(Config):
    model_type: str = Field(
        default="DeepChem/ChemBERTa-77M-MLM", description="Model Description"
    )
    # TODO: Coupling with DataConfig, remove later.
    protein_type: str = Field(default="kinase", description="Type of the protein")


@asset
def initialize_model(
    context: OpExecutionContext, config: MyModelConfig, wandb_resource: WandbResource
) -> ClassificationModel:
    wandb_resource.login()

    model = get_model(model_type=config.model_type)
    context.log.debug(model.tokenizer)

    return model


@asset
def train_model(
    context: OpExecutionContext,
    config: MyModelConfig,
    initialize_model: ClassificationModel,
    train_df_asset: pd.DataFrame,
    valid_df_asset: pd.DataFrame,
) -> ClassificationModel:
    protein_type = config.protein_type
    model_type = config.model_type.replace("/", "_")
    output_dir = f"{protein_type.upper()}_{model_type}"
    # output_dir = f"{protein_type.upper()}_77M_MLM_Shuffle_80_10_10_epoch10"

    context.log.info("Training model ...")
    context.log.debug(f"Output directory: {output_dir}")

    train_model_util(
        model=initialize_model,
        train_df=train_df_asset,
        valid_df=valid_df_asset,
        output_dir=output_dir,
    )

    return initialize_model


@asset
def evaluate_model(
    train_model: ClassificationModel,
    test_df_asset: pd.DataFrame,
) -> None:
    evaluate_model_util(model=train_model, test_df=test_df_asset)


if __name__ == "__main__":
    load_dotenv()

    my_config = MyModelConfig(model_type="DeepChem/ChemBERTa-77M-MLM")

    result = materialize_to_memory(
        assets=[initialize_model],
        run_config=RunConfig(
            ops={
                "initialize_model": my_config,
            },
        ),
        resources={"wandb_resource": WandbResource(apikey=EnvVar("WANDB_API_KEY"))},
    )
    print(result.output_for_node("initialize_model"))  # noqa: T201
