from pathlib import Path
import json
import numpy as np

REPO_DIR = Path(__file__).parent.parent.resolve()
_TRAIN_DATASETS = {}
_TEST_DATASETS = {}

def read_quant_dataset(dataset, regression = False):
    subpath = "regress" if regression else "class"
    y_dtype = np.float64 if regression else np.int32
    with open(REPO_DIR / f"Quant-BnB/dataset/{subpath}/{dataset}.json") as dataset_file:
        data = json.load(dataset_file)

    return (np.array(data["Xtrain"]), np.array(data["Ytrain"], dtype=y_dtype).ravel(), np.array(data["Xtest"]), np.array(data["Ytest"], dtype=y_dtype).ravel())

def get_train_set(dataset: str, task: str):
    if dataset not in _TRAIN_DATASETS:
        Xtrain, Ytrain, Xtest, Ytest = read_quant_dataset(dataset, regression=task=="regression")
        _TRAIN_DATASETS[dataset] = (Xtrain, Ytrain)
        _TEST_DATASETS[dataset] = (Xtest, Ytest)
    
    assert dataset in _TRAIN_DATASETS, "Invalid dataset name"
    return _TRAIN_DATASETS[dataset]

def get_test_set(dataset: str, task: str):
    if dataset not in _TEST_DATASETS:
        Xtrain, Ytrain, Xtest, Ytest = read_quant_dataset(dataset, regression=task=="regression")
        _TRAIN_DATASETS[dataset] = (Xtrain, Ytrain)
        _TEST_DATASETS[dataset] = (Xtest, Ytest)
    
    assert dataset in _TEST_DATASETS, "Invalid dataset name"
    return _TEST_DATASETS[dataset]