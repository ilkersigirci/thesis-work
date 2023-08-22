import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from thesis_work.cv.split import create_folds
from thesis_work.utils.initialization import check_initialization_params
from thesis_work.utils.molnet_dataloader import load_molnet_dataset
from thesis_work.utils.utils import is_valid_smiles

# from thesis_work.chemberta.molnet_dataloader import write_molnet_dataset_for_chemprop

DATA_PATH = Path(__file__).parent.parent.parent / "data"
logger = logging.getLogger(__name__)


def _sample_data(
    df: pd.DataFrame, sample_size: Optional[int] = None, random_state: int = 42
):
    if sample_size is None:
        return df

    if len(df) < sample_size:
        logger.warning(
            f"Sample size is {sample_size} is greater than number of data is {len(df)}."
            f"Hence, sample size is set to {len(df)}."
        )
        sample_size = len(df)

    df = df.sample(n=sample_size, random_state=random_state)

    return df


def _merge_with_vectors(df: pd.DataFrame, vectors: pd.DataFrame):
    """Merges data with vectors"""
    return df.merge(vectors, on="text", how="left")


def load_ataberk(
    subfolder: int,
    compound_name: Optional[str] = None,
    return_vectors: bool = False,
    sample_size: Optional[int] = None,
    random_state: int = 42,
):
    """Loads data from Ataberk's data"""

    check_initialization_params(
        attr=subfolder, accepted_list=["chembl27", "dude", "zinc15"]
    )

    if compound_name is None and subfolder != "zinc15":
        raise ValueError("Compound name must be given.")

    if subfolder == "zinc15":
        compound_name = "zinc15-minor-targets"

    data_root_path = DATA_PATH / "ataberk" / subfolder
    data_smiles_path = data_root_path / "smiles" / f"{compound_name}.csv"

    df = pd.read_csv(data_smiles_path)

    if subfolder == "chembl27":
        check_initialization_params(
            attr=compound_name, accepted_list=["abl1", "renin", "thb"]
        )

        df = df.drop(columns=["ChEMBL"], axis=1)

    elif subfolder == "dude":
        check_initialization_params(
            attr=compound_name, accepted_list=["abl1", "renin", "thb"]
        )

    elif subfolder == "zinc15":
        pass

    if return_vectors is True:
        data_vectors_path = data_root_path / "vectors" / f"{compound_name}.csv"
        vectors_df = pd.read_csv(data_vectors_path)

        df = _merge_with_vectors(df=df, vectors=vectors_df)

    return _sample_data(df, sample_size=sample_size, random_state=random_state)


def load_chembl_30(sample_size: Optional[int] = None, random_state: int = 42):
    # TODO: Change column names

    data_path = DATA_PATH / "chembl_30" / "smiles.tar.xz"
    df = pd.read_csv(data_path, compression="xz", sep="\t")

    return _sample_data(df, sample_size=sample_size, random_state=random_state)


def load_moleculenet(task: str = "bace", sample_size: Optional[int] = None):
    """Loads data of MoleculeNet property prediction classification tasks."""
    # TODO: Change column names for each task

    check_initialization_params(
        attr=task, accepted_list=["bace", "bbbp", "clintox", "hiv", "sider", "tox21"]
    )

    data_path = DATA_PATH / "moleculenet" / f"{task}.csv"
    df = pd.read_csv(data_path)

    return _sample_data(df, sample_size=sample_size)


def load_protein_family(
    protein_type: str = "kinase",
    sample_size: Optional[int] = None,
    random_state: int = 42,
    interacted_only: bool = False,
) -> pd.DataFrame:
    """ "Loads data"""
    data_path = DATA_PATH / "protein_family" / f"{protein_type}.csv"
    df = pd.read_csv(data_path)
    df.columns = ["text", "labels"]

    if interacted_only is True:
        df = df[df["labels"] == 1]

    return _sample_data(df, sample_size=sample_size, random_state=random_state)


def load_protein_family_multiple_interacted(
    protein_types: Optional[List[str]] = None,
    sample_size: int = 3000,
    random_state: int = 42,
    convert_labels: bool = False,
) -> pd.DataFrame:
    """
    Loads interactive compounds from multiple protein types.
    And sample each protein type
    """
    result = pd.DataFrame()

    if protein_types is None:
        protein_types = [
            "gpcr",
            "ionchannel",
            "kinase",
            "nuclearreceptor",
            "protease",
            "transporter",
        ]
    protein_types.sort()

    each_sample_size = sample_size / len(protein_types)

    if isinstance(each_sample_size, float):
        each_sample_size = int(each_sample_size)
        logger.info(
            f"Sample size: {sample_size} is not divisible by number of protein types."
            f"Hence, each protein type will be sampled by {each_sample_size}"
        )

    for protein_type in protein_types:
        data = load_protein_family(
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


def load_protein_family_splits(
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
        df = load_protein_family(protein_type=protein_type)

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

    # Reset index
    df = df.reset_index(drop=True)

    return _sample_data(df, sample_size=sample_size, random_state=random_state)


def save_data(df: pd.DataFrame, name: str, subfolder: str = "result_data"):
    """Saves data"""
    data_path = DATA_PATH / subfolder / f"{name}.csv"
    df.to_csv(data_path, index=False)
