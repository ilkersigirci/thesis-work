"""
- Run script in the background with ssh connection:
nohup python3 -u <script> </dev/null >/dev/null 2>&1 &

nohup /home/ilker/Documents/MyRepos/thesis-work/.venv/bin/python -u /home/ilker/Documents/MyRepos/thesis-work/dagster_pipelines/jobs/finetune.py </dev/null >/dev/null 2>&1 &

- Dagster Config
ops:
  data_asset:
    config:
      protein_type: gpcr
      fixed_cv: true
  initialize_model:
    config:
      protein_type: gpcr
      model_type: DeepChem/ChemBERTa-77M-MLM
      fixed_cv: true
  train_model:
    config:
      protein_type: gpcr
      model_type: DeepChem/ChemBERTa-77M-MLM
      fixed_cv: true

"""
import pandas as pd
from dagster import EnvVar, RunConfig, job
from dotenv import load_dotenv

from dagster_pipelines.assets.chemberta.data import DataConfig, data_asset
from dagster_pipelines.assets.chemberta.model import (
    MyModelConfig,
    evaluate_model,
    initialize_model,
    show_eval_result,
    train_model,
)
from dagster_pipelines.resources.wandb import WandbResource


# FIXME: Evaluate model doesn't run. Hence, pipeline never finishes.
@job
def finetune_job() -> None:
    train_df, valid_df, test_df = data_asset()
    initialized_model = initialize_model()
    trained_model = train_model(initialized_model, train_df, valid_df)
    eval_results = evaluate_model(trained_model, test_df)
    show_eval_result(eval_results)


if __name__ == "__main__":
    load_dotenv()

    protein_type = "kinase"
    # protein_type = "protease"
    # protein_type = "gpcr"

    model_type = "DeepChem/ChemBERTa-77M-MLM"
    fixed_cv = True

    result = finetune_job.execute_in_process(
        run_config=RunConfig(
            {
                "data_asset": DataConfig(protein_type=protein_type, fixed_cv=fixed_cv),
                "initialize_model": MyModelConfig(
                    model_type=model_type, fixed_cv=fixed_cv, protein_type=protein_type
                ),
                "train_model": MyModelConfig(
                    model_type=model_type, protein_type=protein_type, fixed_cv=fixed_cv
                ),
            }
        ),
        resources={"wandb_resource": WandbResource(apikey=EnvVar("WANDB_API_KEY"))},
    )
    assert result.success
    assert isinstance(
        result.output_for_node("data_asset", output_name="train_df_asset"), pd.DataFrame
    )
