import datetime as dt
import json
from typing import Dict, List

from sklearn import metrics
from sklearn.metrics import confusion_matrix

BINARY_CLASSIFICATION_CLASSES = 2


def prec_rec_f1_acc_mcc(
    y_true, y_pred, num_classes: int = 2, save: bool = False
) -> Dict[str, float]:
    performance_threshold_dict = {}

    # TODO: Why do this? To convert them to list?
    # y_true_tmp = []
    # for each_y_true in y_true:
    #     y_true_tmp.append(each_y_true)
    # y_true = y_true_tmp

    # y_pred_tmp = []
    # for each_y_pred in y_pred:
    #     y_pred_tmp.append(each_y_pred)
    # y_pred = y_pred_tmp

    accuracy = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    if num_classes == BINARY_CLASSIFICATION_CLASSES:
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        auroc = metrics.auc(fpr, tpr)
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)

        performance_threshold_dict["Auroc"] = auroc
        performance_threshold_dict["Roc_auc"] = roc_auc
        performance_threshold_dict["TP"] = tp
        performance_threshold_dict["FP"] = fp
        performance_threshold_dict["TN"] = tn
        performance_threshold_dict["FN"] = fn
    else:
        precision = metrics.precision_score(y_true, y_pred, average="weighted")
        recall = metrics.recall_score(y_true, y_pred, average="weighted")
        f1_score = metrics.f1_score(y_true, y_pred, average="weighted")

    performance_threshold_dict["Precision"] = precision
    performance_threshold_dict["Recall"] = recall
    performance_threshold_dict["F1-Score"] = f1_score
    performance_threshold_dict["Accuracy"] = accuracy
    performance_threshold_dict["MCC"] = mcc

    if save is True:
        file_name = f"scores_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(file_name, "w") as f:
            f.write(json.dumps(performance_threshold_dict))

    return performance_threshold_dict


def get_list_of_scores(num_classes: int) -> List[str]:
    general_score_names = ["Precision", "Recall", "F1-Score", "Accuracy", "MCC"]
    binary_classification_score_names = [
        *general_score_names,
        "Auroc",
        "Roc_auc",
        "TP",
        "FP",
        "TN",
        "FN",
    ]

    if num_classes == BINARY_CLASSIFICATION_CLASSES:
        return binary_classification_score_names

    return general_score_names
