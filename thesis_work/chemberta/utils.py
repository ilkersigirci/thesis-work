# from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from sklearn.metrics import accuracy_score, average_precision_score

# from thesis_work.chemberta.molnet_dataloader import write_molnet_dataset_for_chemprop
from thesis_work.chemberta.molnet_dataloader import load_molnet_dataset

DATA_PATH = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent.parent


def load_data_splits(
    protein_type: str = "kinase",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads data and generates folds.

    Args:
        protein_type: Type of the protein.

    Returns:
        train, validation and test data.
    """

    if protein_type == "clintox":
        tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset(
            "clintox", tasks_wanted=None
        )

    else:
        data_path = DATA_PATH / f"{protein_type}_smiles.csv"

        df = pd.read_csv(data_path)
        df.columns = ["text", "labels"]

        df = df.sample(frac=1, random_state=42)

        train_df, valid_df, test_df = np.split(
            # df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))]
            df.sample(frac=1),
            [int(0.8 * len(df)), int(0.9 * len(df))],
        )

    return train_df, valid_df, test_df


def get_model(model_type: str = "DeepChem/ChemBERTa-77M-MLM") -> ClassificationModel:
    model_args = ClassificationArgs(
        evaluate_each_epoch=True,
        evaluate_during_training_verbose=True,
        no_save=True,
        num_train_epochs=10,
        # overwrite_output_dir=True,
        # auto_weights=True, # NOTE: Not working
        # NOTE: Necessary for training outside of Colab
        use_multiprocessing=False,
        # dataloader_num_workers=0,
        # process_count=1,
        use_multiprocessing_for_evaluation=False,
    )

    # Early stopping
    # model_args.use_early_stopping = True
    # model_args.early_stopping_delta = 0.01
    # model_args.early_stopping_metric = "mcc"
    # model_args.early_stopping_metric_minimize = False
    # model_args.early_stopping_patience = 5
    # model_args.evaluate_during_training_steps = 1000

    # model_type = "seyonec/PubChem10M_SMILES_BPE_396_250"  # BPE tokenizer
    # model_type = "seyonec/SMILES_tokenized_PubChem_shard00_160k"  # Custom SMILES tokenizer

    # model_type = "DeepChem/ChemBERTa-10M-MTR"
    # model_type = "DeepChem/ChemBERTa-77M-MLM"

    # if not os.path.exists(f"results/{output_dir}"):
    #     os.makedirs(output_dir)

    # You can set class weights by using the optional weight argument
    return ClassificationModel(
        "roberta",
        model_type,
        args=model_args,
        # use_cuda=False,
    )


def train_model(
    model: ClassificationModel,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: str,
) -> None:
    # Create directory to store model weights (change path accordingly to where you want!)
    # !mkdir BPE_PubChem_10M_ClinTox_run

    # output_dir = "BPE_PubChem_10M_ClinTox_run"
    # output_dir = "SmilesTokenizer_PubChem_10M_ClinTox_run"
    # output_dir = f"{protein_type.upper()}_77M_MLM_Scaffold"

    # FIXME: Wandb output path is NOT correct
    result_output_dir = OUTPUT_PATH / "results" / "experiments" / output_dir
    result_output_dir = str(result_output_dir)

    # Train the model
    global_step, training_details = model.train_model(
        train_df,
        eval_df=valid_df,
        output_dir=result_output_dir,
        args={"wandb_project": output_dir},
    )


def evaluate_model(model, test_df) -> None:
    # FIXME: Takes to much time, 17 min for 6k samples

    # accuracy
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df, acc=accuracy_score
    )

    # ROC-PRC
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df, acc=average_precision_score
    )


def predict_model(model, smiles_mol_list: List[str]):
    predictions, raw_outputs = model.predict(smiles_mol_list)

    return predictions, raw_outputs
