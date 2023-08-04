import logging
import os

import wandb

from thesis_work.chemberta.utils import (  # noqa: F401
    evaluate_model,
    get_model,
    predict_model,
    train_model,
)
from thesis_work.data import load_data_splits

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger(__file__)
transformers_logger.setLevel(logging.WARNING)

# NOTE: Necessary for training outside of Colab
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    wandb.login()

    protein_type = "kinase"
    output_dir = f"{protein_type.upper()}_77M_MLM_Shuffle_80_10_10_epoch10"

    train_df, valid_df, test_df = load_data_splits(protein_type=protein_type)
    model = get_model()

    # print(model.tokenizer)

    train_model(
        model=model, train_df=train_df, valid_df=valid_df, output_dir=output_dir
    )
    # evaluate_model(model=model, test_df=test_df)

    # Lets input a molecule with a toxicity value of 1
    # mol = "C1=C(C(=O)NC(=O)N1)F"
    # mol = "Nc1nc(NC2CC2)c2ncn([C@H]3C=C[C@@H](CO)C3)c2n1"

    # predict_model(model=model, smiles_mol_list=[mol])
