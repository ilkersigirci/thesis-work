import os

import pandas as pd
import wandb


def get_project_summary(project_name: str) -> pd.DataFrame:
    WANDB_USER_NAME = os.environ["WANDB_USER_NAME", None]

    if WANDB_USER_NAME is None:
        raise ValueError("WANDB_USER_NAME environment variable is not set.")

    api = wandb.Api()
    runs = api.runs(WANDB_USER_NAME + "/" + project_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    return runs_df


def get_metric_from_project(project_name: str, metric: str) -> dict:
    runs_df = get_project_summary(project_name=project_name)

    run_names = runs_df["name"].apply(
        lambda x: x.replace("\n", "").strip().upper().replace(" ", "")
    )
    metric_scores = runs_df["summary"].apply(lambda x: x.get(metric, None))

    if metric_scores.isna().sum() > 0:
        raise ValueError(f"Metric {metric} is not found in the project.")

    return dict(zip(run_names, metric_scores))
