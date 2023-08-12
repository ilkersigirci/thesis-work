from pathlib import Path

import numpy as np
import pandas as pd

from molbert_repo.molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

TRAINED_MODEL_FOLDER_PATH = Path(__file__).parent.parent.parent / "models" / "molbert"
CHECKPOINT_MODEL_PATH = TRAINED_MODEL_FOLDER_PATH / "checkpoints/last.ckpt"


# FIXME: Not working
def get_model_descriptors(smiles_series: pd.Series) -> np.array:
    featurizer = MolBertFeaturizer(CHECKPOINT_MODEL_PATH)
    features, masks = featurizer.transform(["C"])

    return features
