"""Dagster pipelines for thesis-work."""
import warnings

from dagster import Definitions, EnvVar, ExperimentalWarning

from .jobs.finetune import finetune_job
from .resources.wandb import WandbResource

warnings.filterwarnings("ignore", category=ExperimentalWarning)

from .assets import chemberta_assets, playground_assets

JOBS = [finetune_job]
SCHEDULES = []
SENSORS = []

# ASSETS = [chemberta_assets, playground_assets] # FIXME: Not working
ASSETS = chemberta_assets + playground_assets

RESOURCES = {
    "wandb_resource": WandbResource(apikey=EnvVar("WANDB_API_KEY")),
}

defs = Definitions(
    assets=ASSETS,
    jobs=JOBS,
    schedules=SCHEDULES,
    sensors=SENSORS,
    resources=RESOURCES,
)
