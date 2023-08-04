from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from thesis_work.cv.split import create_folds
from thesis_work.molnet_dataloader import load_molnet_dataset

# from thesis_work.chemberta.molnet_dataloader import write_molnet_dataset_for_chemprop

DATA_PATH = Path(__file__).parent / "data"


def load_data(protein_type: str = "kinase") -> pd.DataFrame:
    """ "Loads data"""
    data_path = DATA_PATH / f"{protein_type}_smiles.csv"
    df = pd.read_csv(data_path)
    df.columns = ["text", "labels"]

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
