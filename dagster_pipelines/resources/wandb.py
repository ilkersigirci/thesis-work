from dagster import ConfigurableResource

import wandb


class WandbResource(ConfigurableResource):
    apikey: str

    def login(self) -> None:
        wandb.login(key=self.apikey)
