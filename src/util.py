from pathlib import Path
import json
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()

def read_quant_dataset(dataset, regression = False):
    subpath = "regress" if regression else "class"
    y_dtype = np.float64 if regression else np.int32
    with open(SCRIPT_DIR / f"../Quant-BnB/dataset/{subpath}/{dataset}.json") as dataset_file:
        data = json.load(dataset_file)

    return (np.array(data["Xtrain"]), np.array(data["Ytrain"], dtype=y_dtype), np.array(data["Xtest"]), np.array(data["Ytest"], dtype=y_dtype))
