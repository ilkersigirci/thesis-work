import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from thesis_work.cv.split import create_folds
from thesis_work.molnet_dataloader import load_molnet_dataset
from thesis_work.utils import is_valid_smiles

# from thesis_work.chemberta.molnet_dataloader import write_molnet_dataset_for_chemprop

DATA_PATH = Path(__file__).parent.parent / "data"
logger = logging.getLogger(__name__)


# TODO:
def transform_bbbp():
    pass


def load_data(
    protein_type: str = "kinase",
    subfolder: str = "original",
    sample_size: Optional[int] = None,
    random_state: int = 42,
    interacted_only: bool = False,
) -> pd.DataFrame:
    """ "Loads data"""
    data_path = DATA_PATH / subfolder / f"{protein_type}.csv"
    df = pd.read_csv(data_path)
    df.columns = ["text", "labels"]

    if interacted_only is True:
        df = df[df["labels"] == 1]

    if sample_size is not None:
        if len(df) < sample_size:
            logger.warning(
                f"Sample size is {sample_size} is greater than number of data is {len(df)}."
            )

        df = df.sample(n=sample_size, random_state=random_state)

    return df


def save_data(df: pd.DataFrame, name: str, subfolder: str = "result_data"):
    """Saves data"""
    data_path = DATA_PATH / subfolder / f"{name}.csv"
    df.to_csv(data_path, index=False)


def load_mixed_interacted_compounds(
    protein_types: Optional[List[str]] = None,
    each_sample_size: int = 1000,
    random_state: int = 42,
    convert_labels: bool = False,
) -> pd.DataFrame:
    """
    Loads interactive compounds from multiple protein types.
    And sample each protein type
    """
    result = pd.DataFrame()

    if protein_types is None:
        protein_types = ["gpcr", "kinase", "protease"]

    protein_types.sort()

    for protein_type in protein_types:
        data = load_data(
            protein_type=protein_type,
            sample_size=each_sample_size,
            random_state=random_state,
            interacted_only=True,
        )
        # data["labels"] = index
        # data["labels_protein"] = protein_type
        data["labels"] = protein_type
        result = pd.concat([result, data], axis=0, ignore_index=True)

    if convert_labels is True:
        # result["labels_protein"] = result["labels"].astype("category").cat.codes
        result["labels_protein"] = pd.factorize(result["labels"])[0]

    return result


def load_related_work(
    sample_size: Optional[int] = None,
    random_state: int = 42,
):
    """Loads data from related work"""
    data_path = DATA_PATH / "related_work" / "unbiased" / "compound_annotation.csv"
    df = pd.read_csv(data_path, usecols=["SMILES"])
    df = df.rename(columns={"SMILES": "text"})

    # FIXME: Df has no labels. This is a temporary fix.
    df["labels"] = 0

    df = df[df["text"].apply(is_valid_smiles)]

    if sample_size is not None:
        if len(df) < sample_size:
            logger.warning(
                f"Sample size is {sample_size} is greater than number of data is {len(df)}."
            )

        df = df.sample(n=sample_size, random_state=random_state)

    return df


def load_data_splits(
    protein_type: str = "kinase", fixed_cv: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Loads data and generates folds.

    Args:
        protein_type: Type of the protein.
        fixed_cv: Whether to use fixed CV. If True, then the last fold is used as
         test set and no validation set is used.

    Returns:
        train, validation and test data.
    """

    if protein_type == "clintox":
        tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset(
            "clintox", tasks_wanted=None
        )

    else:
        df = load_data(protein_type=protein_type)

        if fixed_cv is True:
            len_data = len(df)
            fold_list = create_folds(length=len_data)
            train_indices = [item for sublist in fold_list[:5] for item in sublist]
            test_indices = fold_list[5]

            train_df = df.iloc[train_indices]
            valid_df = None
            test_df = df.iloc[test_indices]

        else:
            df = df.sample(frac=1, random_state=42)

            train_df, valid_df, test_df = np.split(
                # df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))]
                df.sample(frac=1),
                [int(0.8 * len(df)), int(0.9 * len(df))],
            )

    return train_df, valid_df, test_df
