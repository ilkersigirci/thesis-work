from typing import Optional, Tuple

import pandas as pd
from dagster import (
    AssetOut,
    Config,
    OpExecutionContext,
    RunConfig,
    materialize_to_memory,
    multi_asset,
)
from pydantic import Field

from thesis_work.chemberta.utils import load_data_splits


class DataConfig(Config):
    protein_type: str = Field(default=..., description="Type of the protein")
    fixed_cv: bool = Field(default=False, description="Whether to use fixed CV")


@multi_asset(
    outs={
        "train_df_asset": AssetOut(),
        "valid_df_asset": AssetOut(),
        "test_df_asset": AssetOut(),
    }
)
def data_asset(
    context: OpExecutionContext,
    config: DataConfig,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    train_df, valid_df, test_df = load_data_splits(
        protein_type=config.protein_type, fixed_cv=config.fixed_cv
    )

    if config.fixed_cv is True:
        assert valid_df is None

    log_info = f"train_df.shape: {train_df.shape} test_df.shape: {test_df.shape} "
    if valid_df is not None:
        log_info = log_info + f"valid_df.shape: {valid_df.shape}"

    context.log.info(log_info)

    return train_df, valid_df, test_df


if __name__ == "__main__":
    my_config = DataConfig(protein_type="kinase", fixed_cv=True)

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
