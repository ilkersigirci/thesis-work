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


if __name__ == "__main__":
    apply_all(family="kinase", compound_count=66310)
