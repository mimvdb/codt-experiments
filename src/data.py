import json
import numpy as np
import pandas as pd
from functools import cache
from sklearn.model_selection import KFold, train_test_split
from src.util import REPO_DIR

DATASETS_CLASSIFICATION = ["avila", "bank", "bean", "bidding", "eeg", "fault", "htru", "magic", "occupancy", "page", "raisin", "rice", "room", "segment", "skin", "wilt"]
# Exclude carbon and query2 regression sets from Quant-BnB, since they are multivariate.
DATASETS_REGRESSION = ["casp", "concrete", "energy", "fish", "gas", "grid", "news", "qsar", "query1"]

def read_quant_dataset(dataset, regression = False):
    subpath = "regress" if regression else "class"
    y_dtype = np.float64 if regression else np.int32
    with open(REPO_DIR / f"Quant-BnB/dataset/{subpath}/{dataset}.json") as dataset_file:
        data = json.load(dataset_file)

    return (np.array(data["Xtrain"]), np.array(data["Ytrain"], dtype=y_dtype).ravel(), np.array(data["Xtest"]), np.array(data["Ytest"], dtype=y_dtype).ravel())

@cache
def get_dataset(dataset: str, task: str):
    path = REPO_DIR / "datasets" / task / f"{dataset}.csv"
    assert path.exists(), f"Dataset {dataset} does not exist for task {task}"
    df = pd.read_csv(path, sep=" ", header=None)
    X = df[df.columns[1:]].to_numpy()
    y = df[0].to_numpy()
    return (X, y)

@cache
def get_test_indices(dataset: str, task: str, split: str):
    path = REPO_DIR / "datasets" / task / dataset / f"{split}.csv"
    assert path.exists(), f"Index set {split} does not exist for dataset {dataset} of task {task}"
    indices = pd.read_csv(path, sep=" ", header=None).to_numpy().ravel()
    return indices

def quant_to_csv():
    for cdata in DATASETS_CLASSIFICATION:
        Xtrain, Ytrain, Xtest, Ytest = read_quant_dataset(cdata, regression=False)
        X, y = (np.concatenate([Xtrain, Xtest]), np.concatenate([Ytrain, Ytest]))
        df = pd.DataFrame(X)
        df.insert(0, "0", y)
        path = REPO_DIR / "datasets" / "classification" / f"{cdata}.csv"
        path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(path, sep=" ", header=False, index=False)
    for rdata in DATASETS_REGRESSION:
        Xtrain, Ytrain, Xtest, Ytest = read_quant_dataset(rdata, regression=True)
        X, y = (np.concatenate([Xtrain, Xtest]), np.concatenate([Ytrain, Ytest]))
        df = pd.DataFrame(X)
        df.insert(0, "y", y)
        path = REPO_DIR / "datasets" / "regression" / f"{rdata}.csv"
        path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(path, sep=" ", header=False, index=False)

def generate_index_sets():
    for rdata in DATASETS_REGRESSION:
        X, y = get_dataset(rdata, "regression")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, (_, test_indices) in enumerate(kf.split(X, y)):
            path = REPO_DIR / "datasets" / "regression" / rdata / f"{i}.csv"
            path.parent.mkdir(exist_ok=True)
            pd.Series(test_indices).to_csv(path, header=False, index=False)
