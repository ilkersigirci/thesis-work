"""
python fingerprint.py --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_fingerprint.csv

python fingerprint.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_fingerprint.csv


"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDataLoader, get_data_from_smiles
from chemprop.train.molecule_fingerprint import model_fingerprint
from chemprop.utils import load_checkpoint

from thesis_work.utils import check_device

# from chemprop.train.molecule_fingerprint import molecule_fingerprint

logger = logging.getLogger(__name__)

TRAINED_MODEL_FOLDER_PATH = Path(__file__).parent.parent.parent / "models" / "chemprop"
CHECKPOINT_MODEL_PATH = (
    TRAINED_MODEL_FOLDER_PATH / "multi_task_subfamily_dmpnn_25/fold_0/model_0/model.pt"
)


def load_model(path: str, device: str = "cuda") -> torch.nn.Module:
    """Loads a model checkpoint."""
    check_device(device=device)
    device = torch.device(device)

    model = load_checkpoint(path=path, device=device)

    return model


def get_model_descriptors(smiles_series: pd.Series) -> np.array:
    """Calculates and returns model vector embedding for given smiles list.

    Args:
        smiles_series: Smiles data.

    Returns:
        Model vector embeddings.

    TODO:
        - MAKE THIS FAST !!
        - Use `molecule_fingerprint`, since it handles everything.
        The only problem is that, it reads from file and write to file.
    """

    model = load_model(path=CHECKPOINT_MODEL_PATH)
    smiles_list = smiles_series.tolist()

    data = get_data_from_smiles(smiles=[[smiles] for smiles in smiles_list])

    data_loader = MoleculeDataLoader(dataset=data, batch_size=64, num_workers=8)

    descriptors = model_fingerprint(
        model=model,
        data_loader=data_loader,
        fingerprint_type="MPN",
        disable_progress_bar=False,
    )

    return descriptors


if __name__ == "__main__":
    smiles_series = pd.Series(["CCO", "CCN", "CCC"])
    descriptors = get_model_descriptors(smiles_series=smiles_series)
    print(descriptors)  # noqa: T201
