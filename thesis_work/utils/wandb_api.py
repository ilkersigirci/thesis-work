import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import wandb

from thesis_work.utils.initialization import check_initialization_params

# import seaborn as sns

logger = logging.getLogger(__name__)


def get_project_summary(project_name: str) -> pd.DataFrame:
    """Only returns last logged values"""

    WANDB_USER_NAME = os.environ.get("WANDB_USER_NAME", None)

    if WANDB_USER_NAME is None:
        raise ValueError("WANDB_USER_NAME environment variable is not set.")

    api = wandb.Api()
    runs = api.runs(WANDB_USER_NAME + "/" + project_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters. We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    runs_df["name"] = runs_df["name"].apply(
        lambda x: x.replace("\n", "").strip().upper().replace(" ", "")
    )

    return runs_df


def get_processed_project_summary(project_name: str) -> pd.DataFrame:
    metric_x_index_name_mapping = {
        "K-MEANS": "n_clusters",
        "AGGLOMERATIVE": "n_clusters",
        "BUTINA": "threshold",
        "HDBSCAN": "min_cluster_size",
    }

    runs_df = get_project_summary(project_name=project_name)

    runs_df["summary"] = runs_df["summary"].apply(
        lambda x: {
            k: v
            for k, v in x.items()
            if k
            not in [
                "_wandb",
                "_timestamp",
                "_step",
                "Original Labels",
                "Cluster Calculated Labels",
            ]
        }
    )

    runs_df["metric_x_index_name"] = runs_df["name"].apply(
        lambda x: metric_x_index_name_mapping[x.split("_")[0]]
    )

    return runs_df


def get_project_history(project_name: str):
    """Returns all logged values

    NOTE: The default history method samples the metrics to a fixed number of samples
    (the default is 500, you can change this with the samples __ argument).
    If you want to export all of the data on a large run, you can use the run.scan_history() method.
    https://docs.wandb.ai/guides/track/public-api-guide#sampling
    """
    WANDB_USER_NAME = os.environ.get("WANDB_USER_NAME", None)

    if WANDB_USER_NAME is None:
        raise ValueError("WANDB_USER_NAME environment variable is not set.")

    api = wandb.Api()
    runs = api.runs(WANDB_USER_NAME + "/" + project_name)
    run_history_list = []

    for run in runs:
        run_history = run.history()
        run_history = run_history.drop[
            ["_timestamp", "_step", "Original Labels", "Cluster Calculated Labels"]
        ]

        run_history_list.append(run_history)

    return pd.concat(run_history_list, ignore_index=True, sort=False)


def get_metric_from_project(
    project_name: str,
    metric: str,
    metric_x_index_name: str,
    filter_column_contains_substring: Optional[str] = None,
) -> pd.DataFrame:
    WANDB_USER_NAME = os.environ.get("WANDB_USER_NAME", None)

    if WANDB_USER_NAME is None:
        raise ValueError("WANDB_USER_NAME environment variable is not set.")

    api = wandb.Api()
    runs = api.runs(WANDB_USER_NAME + "/" + project_name)

    run_history_list = []
    for run in runs:
        run_history = run.history()
        run_history_columns = run_history.columns

        if (
            metric not in run_history_columns
            or metric_x_index_name not in run_history_columns
        ):
            logger.debug(
                f"Skipping since necessary columns are not found in the run history, {run.name}."
            )
            continue

        run_history = (
            run_history[[metric_x_index_name, metric]].dropna().reset_index(drop=True)
        )

        if run_history.empty:
            logger.debug(f"Skipping because of empty run history, {run.name}.")
            continue

        run_history["name"] = run.name

        if filter_column_contains_substring is not None:
            run_history = run_history[
                run_history["name"].str.contains(
                    filter_column_contains_substring, case=False
                )
            ]

        if run_history.empty:
            logger.debug(
                f"Skipping because of empty run history filtered with {filter_column_contains_substring}, {run.name}."
            )
            continue

        run_history[metric] = run_history[metric].astype(float).round(3)
        run_history[metric_x_index_name] = run_history[metric_x_index_name].astype(int)

        run_history_list.append(run_history)

    if not run_history_list:
        raise ValueError(f"Metric {metric} is not found in the project.")

    return pd.concat(run_history_list, ignore_index=True, sort=False)


def _process_figure_legend(ax, remove_string: str = ""):
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace(remove_string, "") for label in labels]

    # For ChemBERTa suffix
    chemberta_suffix = "-77M-MTR"
    labels = [label.replace(chemberta_suffix, "") for label in labels]
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="lower right")


def plot_metric_from_project(  # noqa: PLR0913
    df: pd.DataFrame,
    # project_name: str,
    metric: str,
    metric_x_index_name: str,
    filter_column_contains_substring: str = "",
    # hue: str = None,
    # palette: str = "tab10",
    show_title: bool = False,
    save_path: Optional[str] = None,
    method: str = "matplotlib",
):
    check_initialization_params(
        attr=method, accepted_list=["matplotlib", "seaborn", "plotly"]
    )
    # df.assign(base=df['name'].apply(lambda x: 16 if '16' in x else 32))

    # For ChemBERTa
    df["name"] = df["name"].apply(lambda x: x.replace("-77M-MTR", ""))

    if method == "plotly":
        fig = px.line(
            df,
            x=metric_x_index_name,
            y=metric,
            color="name",
            title=f"{metric} score",
            markers=False,
            width=1280,
            height=720,
            # facet_col="base",
        )

        if show_title is False:
            fig.update_layout(title_text=None)

        if save_path is not None:
            fig.write_image(save_path)

        return fig

    fig, ax = plt.subplots(figsize=(12, 7))

    if method == "matplotlib":
        for name, group in df.groupby("name"):
            ax.plot(group[metric_x_index_name], group[metric], label=name)

    elif method == "seaborn":
        sns.lineplot(
            data=df,
            x=metric_x_index_name,
            y=metric,
            hue="name",
            ax=ax,
            markers=False,
            style="name",
        )

    ax.set_xlabel(metric_x_index_name)
    ax.set_ylabel(metric)
    ax.legend(loc="lower right")
    ax.grid(False)

    if show_title is True:
        ax.set_title(f"{metric} score")

    plt.close(fig)

    _process_figure_legend(ax=ax, remove_string=filter_column_contains_substring)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

        size = os.path.getsize(save_path)
        logger.debug(f"The size of the saved figure is {size} bytes.")

    return fig
