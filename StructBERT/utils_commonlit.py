import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def create_folds(data: pd.DataFrame, num_splits: int, seed: int):
    data["kfold"] = -1
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)
    kf = StratifiedKFold(n_splits=num_splits)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    data = data.drop("bins", axis=1)
    return data

if __name__ == "__main__":
    NFOLDS = 5
    SEED = 1000
    df_train = pd.read_csv("data/commonlit/train.csv")
    df_train = create_folds(df_train, num_splits=NFOLDS, seed=SEED)
    print(df_train.shape)
    print(df_train.head(2))

    df_train = df_train.rename(columns={"target":"label"})

    for fold in range(NFOLDS):
        os.makedirs(f"data/commonlit/fold_{fold}", exist_ok=True)
        train_fold = df_train.query("kfold!=@fold").reset_index(drop=True)
        valid_fold = df_train.query("kfold==@fold").reset_index(drop=True)
        print(train_fold.shape); print(valid_fold.shape)
        train_fold[["excerpt","label"]].to_csv(f"data/commonlit/fold_{fold}/train.tsv", sep="\t", index=False)
        valid_fold[["excerpt","label"]].to_csv(f"data/commonlit/fold_{fold}/valid.tsv", sep="\t", index=False)
        print("Saving Fold", fold)
