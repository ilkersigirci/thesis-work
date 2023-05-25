# from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from sklearn.metrics import accuracy_score, average_precision_score

# from thesis_work.chemberta.molnet_dataloader import write_molnet_dataset_for_chemprop
from thesis_work.chemberta.molnet_dataloader import load_molnet_dataset
from thesis_work.cv.split import create_folds

DATA_PATH = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent.parent.parent


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
        data_path = DATA_PATH / f"{protein_type}_smiles.csv"

        df = pd.read_csv(data_path)
        df.columns = ["text", "labels"]

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
    valid_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> None:
    # Create directory to store model weights (change path accordingly to where you want!)
    # !mkdir BPE_PubChem_10M_ClinTox_run

    if output_dir is None:
        output_dir = "Untitled_Run"

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


def evaluate_model(
    model: ClassificationModel, test_df: pd.DataFrame
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # FIXME: Takes to much time, 17 min for 6k samples
    # TODO: Run eval on the same model for multiple times to see if the results are consistent

    # accuracy
    result_acc, model_outputs_acc, wrong_predictions_acc = model.eval_model(
        test_df, acc=accuracy_score
    )

    # ROC-PRC
    result_avs, model_outputs_avs, wrong_predictions_avs = model.eval_model(
        test_df, acc=average_precision_score
    )

    result_acc_wandb = {f"ACC_{key}": value for key, value in result_acc.items()}
    result_avs_wandb = {f"AVS_{key}": value for key, value in result_avs.items()}
    resul_all_wandb = {**result_acc_wandb, **result_avs_wandb}

    wandb.log(resul_all_wandb)

    # mcc = result_avs["mcc"]
    # tp = result_avs["tp"]
    # tn = result_avs["tn"]
    # fp = result_avs["fp"]
    # fn = result_avs["fn"]
    # auroc = result_avs["auroc"]
    # auprc = result_avs["auprc"]
    # acc = result_acc["acc"]
    # eval_loss = result_acc["eval_loss"]

    return result_acc, result_avs


def predict_model(model: ClassificationModel, smiles_mol_list: List[str]):
    predictions, raw_outputs = model.predict(smiles_mol_list)

    return predictions, raw_outputs
