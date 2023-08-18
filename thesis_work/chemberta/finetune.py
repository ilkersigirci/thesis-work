import logging
import os

import wandb

from thesis_work.chemberta.utils import (
    evaluate_model,
    get_model,
    predict_model,
    train_model,
)
from thesis_work.utils.data import load_protein_family_splits

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger(__file__)
transformers_logger.setLevel(logging.WARNING)

# NOTE: Necessary for training outside of Colab
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    wandb.login()

    method = "finetune"
    # method = "evaluate"
    # method = "predict"

    protein_type = "kinase"
    # protein_type = "gpcr"
    # protein_type = "protease"

    output_dir = f"{protein_type.upper()}_77M_MLM_Shuffle_80_10_10_epoch10"

    train_df, valid_df, test_df = load_protein_family_splits(protein_type=protein_type)
    model = get_model()

    if method == "finetune":
        train_model(
            model=model, train_df=train_df, valid_df=valid_df, output_dir=output_dir
        )

    elif method == "evaluate":
        evaluate_model(model=model, test_df=test_df)

    elif method == "predict":
        mol = "C1=C(C(=O)NC(=O)N1)F"
        mol = "Nc1nc(NC2CC2)c2ncn([C@H]3C=C[C@@H](CO)C3)c2n1"

        predict_model(model=model, smiles_mol_list=[mol])
