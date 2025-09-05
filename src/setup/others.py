import itertools
from typing import List
from src.data import DATASETS_REGRESSION, DATASETS_CLASSIFICATION
from src.methods import RunParams

TIMEOUT = 30 * 60 # timeout in seconds


def setup_others() -> List[RunParams]:
    return setup_others_class() + setup_others_reg()


def setup_others_reg() -> List[RunParams]:
    runs = []
    depths = [2, 3, 4]
    methods = ["quantbnb"]
    for m, d, data in itertools.product(methods, depths, DATASETS_REGRESSION):
        runs.append(
            RunParams(
                method=m,
                task="regression",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                memory_limit=7900 * 1024 * 1024
            )
        )
    return runs

def setup_others_class() -> List[RunParams]:
    runs = []
    depths = [2, 3, 4]
    methods = ["contree", "quantbnb"]
    for m, d, data in itertools.product(methods, depths, DATASETS_CLASSIFICATION):
        runs.append(
            RunParams(
                method=m,
                task="classification",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                memory_limit=7900 * 1024 * 1024
            )
        )
    return runs