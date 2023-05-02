from typing import Tuple

import pandas as pd
from dagster import AssetOut, Config, RunConfig, materialize_to_memory, multi_asset
from pydantic import Field

from thesis_work.chemberta.utils import load_data_splits


class DataConfig(Config):
    protein_type: str = Field(default="kinase", description="Type of the protein")


@multi_asset(
    outs={
        "train_df_asset": AssetOut(),
        "valid_df_asset": AssetOut(),
        "test_df_asset": AssetOut(),
    }
)
def data_asset(config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, valid_df, test_df = load_data_splits(protein_type=config.protein_type)

    return train_df, valid_df, test_df


if __name__ == "__main__":
    my_config = DataConfig(protein_type="kinase")

    result = materialize_to_memory(
        assets=[data_asset],
        run_config=RunConfig(
            ops={
                "data_asset": my_config,
            },
        ),
    )

    train_df = result.output_for_node("data_asset", output_name="train_df_asset")
    valid_df = result.output_for_node("data_asset", output_name="valid_df_asset")
    test_df = result.output_for_node("data_asset", output_name="test_df_asset")

    print(train_df.shape, valid_df.shape, test_df.shape)  # noqa: T201
