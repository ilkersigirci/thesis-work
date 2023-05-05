"""train_test_5_fold"""
import os
import random


def create_folds(length: int):
    random.seed(69)
    n = 6

    dtiList = list(range(0, length))

    random.shuffle(dtiList)
    newList = [dtiList[i::n] for i in range(n)]

    return newList


def write_folds(FileName, dti_list):
    with open(FileName, "w") as f:
        f.write(str(dti_list))


def apply_all(family: str, compound_count: int) -> None:
    # TODO: Find compound_count from family file

    fold_list = create_folds(length=compound_count)
    os.makedirs(
        os.path.dirname("training_files/" + family + "/data/folds/"), exist_ok=True
    )

    train_fold_list = []
    for i in range(5):
        train_fold_list.append(fold_list[i])

    write_folds(
        "training_files/" + family + "/data/folds/train_fold_setting1.txt",
        train_fold_list,
    )
    write_folds(
        "training_files/" + family + "/data/folds/test_fold_setting1.txt", fold_list[5]
    )


def get_kfold_experiment_indices(length: int):
    fold_count = 6
    all_folds = create_folds(length=length)

    all_train_fold_list = all_folds[: fold_count - 1]
    test_fold_list = all_folds[fold_count - 1]

    experiment_indices = []

    for i in range(fold_count - 1):
        test_df = test_fold_list

        valid_df = all_train_fold_list[i]

        # Remove i from the list and flatten the list
        train_df = [
            item
            for sublist in all_train_fold_list[:i] + all_train_fold_list[i + 1 :]
            for item in sublist
        ]

        experiment_indices.append((train_df, valid_df, test_df))

    return experiment_indices


if __name__ == "__main__":
    # apply_all(family="kinase", compound_count=66310)
    apply_all(family="kinase", compound_count=50)
