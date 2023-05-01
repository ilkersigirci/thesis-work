# from __future__ import annotations  # FIXME: Not working in dagster

from dagster import Config, EnvVar, RunConfig, asset, materialize_to_memory
from pydantic import Field
from simpletransformers.classification import ClassificationModel

from dagster_pipelines.resources.wandb import WandbResource
from thesis_work.chemberta.utils import get_model


class MyModelConfig(Config):
    model_type: str = Field(
        default="DeepChem/ChemBERTa-77M-MLM", description="Model Description"
    )


@asset
def model_asset(
    config: MyModelConfig, wandb_resource: WandbResource
) -> ClassificationModel:
    wandb_resource.login()

    return get_model(model_type=config.model_type)


if __name__ == "__main__":
    my_config = MyModelConfig(model_type="DeepChem/ChemBERTa-77M-MLM")

    result = materialize_to_memory(
        assets=[model_asset],
        run_config=RunConfig(
            ops={
                "model_asset": my_config,
            },
        ),
        resources={"wandb_resource": WandbResource(apikey=EnvVar("WANDB_API_KEY"))},
    )

    print(result.output_for_node("model_asset"))  # noqa: T201
