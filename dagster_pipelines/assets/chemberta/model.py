# FIXME: Not working in dagster:
# from __future__ import annotations


from copy import deepcopy
from typing import Any, Dict, Optional

import pandas as pd
import wandb
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

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
    protein_type: str = Field(default=..., description="Type of the protein")
    fixed_cv: bool = Field(default=False, description="Whether to use fixed CV")


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
    valid_df_asset: Optional[pd.DataFrame] = None,
) -> ClassificationModel:
    # TODO: Make output dir configurable

    protein_type = config.protein_type
    model_type = config.model_type.replace("/", "_")
    # output_dir = f"{model_type[-7:]}_{protein_type.upper()}_CV_20Epochs"
    output_dir = f"{model_type[-7:]}_{protein_type.upper()}_10Epochs"
    # output_dir = f"{protein_type.upper()}_{model_type}"
    # output_dir = f"{protein_type.upper()}_77M_MLM_Shuffle_80_10_10_epoch10"

    if valid_df_asset is None and config.fixed_cv is True:
        context.log.info("Training model with cv...")
        context.log.info(f"Output directory: {output_dir}")

        kfold = KFold(n_splits=5, shuffle=False)
        model = initialize_model
        results = []

        # TODO: Reinitialize model for each fold
        # TODO: average_precision_score can be added to eval
        for i, (train_index, val_index) in enumerate(kfold.split(train_df_asset)):
            context.log.info(f"Fold: {i+1}")

            model = deepcopy(initialize_model)
            model.args.overwrite_output_dir = True
            train_df_fold = train_df_asset.iloc[train_index]
            valid_df_fold = train_df_asset.iloc[val_index]
            train_model_util(
                model=model,
                train_df=train_df_fold,
                valid_df=valid_df_fold,
                output_dir=output_dir,
            )

            result, model_outputs, wrong_predictions = model.eval_model(
                valid_df_fold, acc=accuracy_score
            )
            fold_accuracy = result["acc"]
            results.append(fold_accuracy)

            wandb.log({"fold_accuracy": fold_accuracy})
            context.log.info(f"Accuracy: {fold_accuracy}")

        mean_accuracy = sum(results) / len(results)

        wandb.log({"fold_mean_accuracy": mean_accuracy})
        context.log.info(f"Mean Precision Accuracy: {mean_accuracy}")

        initialize_model = model

    else:
        context.log.info("Training model without cv...")
        context.log.info(f"Output directory: {output_dir}")

        train_model_util(
            model=initialize_model,
            train_df=train_df_asset,
            valid_df=valid_df_asset,
            output_dir=output_dir,
        )

    return initialize_model


@asset
def evaluate_model(
    context: OpExecutionContext,
    train_model: ClassificationModel,
    test_df_asset: pd.DataFrame,
) -> Dict[str, Any]:
    context.log.info("Evaluating model...")
    result_acc, result_avs = evaluate_model_util(
        model=train_model, test_df=test_df_asset
    )

    return result_acc


@asset
def show_eval_result(
    context: OpExecutionContext, evaluate_model: Dict[str, Any]
) -> None:
    context.log.info(f"Accuracy: {evaluate_model['acc']}")
    # context.log.info(f"Average Precision Score: {evaluate_model['aps']}")


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
