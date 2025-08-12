import itertools
from typing import List
from src.data import DATASETS_REGRESSION
from src.methods import RunParams

test_sets = ["0", "1", "2", "3", "4"]
timeout = 4 * 60 * 60 - 5 * 60 # 4h - 5mins

def setup_cart() -> List[RunParams]:
    runs = []
    depths = [2, 3, None]  # None means no max depth.
    for d, data, test_set in itertools.product(depths, DATASETS_REGRESSION, test_sets):
        runs.append(
            RunParams(
                method="cart",
                task="regression",
                timeout=timeout,
                dataset=data,
                max_depth=d,
                test_set=test_set,
                tune=True
            )
        )
    return runs

def setup_codt() -> List[RunParams]:
    runs = []
    depths = [3, 4]
    for d, data, test_set in itertools.product(depths, DATASETS_REGRESSION, test_sets):
        runs.append(
            RunParams(
                method="codt",
                task="regression",
                timeout=timeout,
                dataset=data,
                max_depth=d,
                test_set=test_set,
                tune=True,
                memory_limit=7900 * 1024 * 1024
            )
        )
    return runs

def setup_generalisation() -> List[RunParams]:
    runs = []
    # runs.extend(setup_cart())
    runs.extend(setup_codt())
    return runs
