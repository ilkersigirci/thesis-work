from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from sklearn.metrics import accuracy_score, average_precision_score

OUTPUT_PATH = Path(__file__).parent.parent.parent


def get_model(model_type: str = "DeepChem/ChemBERTa-77M-MLM") -> ClassificationModel:
    """
    TODO:
        - `model_type` parameter is actually `model_name`. Change it!
    """
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
        model_type="roberta",
        model_name=model_type,
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
