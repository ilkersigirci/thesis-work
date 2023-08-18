import wandb
from dagster import ConfigurableResource


class WandbResource(ConfigurableResource):
    apikey: str

    def login(self) -> None:
        wandb.login(key=self.apikey)
