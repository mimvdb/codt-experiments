import itertools
from typing import List
from src.methods import RunParams


def setup_quantbnb_regression() -> List[RunParams]:
    runs = []
    datasets = ["casp", "concrete", "energy", "fish", "gas", "grid", "news", "qsar", "query1"]
    depths = [2, 3]
    test_sets = ["", "0", "1", "2", "3", "4"]
    for d, data, test_set in itertools.product(depths, datasets, test_sets):
        runs.append(
            RunParams(
                method="quantbnb",
                task="regression",
                timeout=10*60,
                dataset=data,
                max_depth=d,
                test_set=test_set,
                tune=False if test_set == "" else True
            )
        )
    return runs

def setup_quantbnb_classification() -> List[RunParams]:
    runs = []
    depths = [2, 3]
    datasets = ["avila", "bank", "bean", "bidding", "eeg", "fault", "htru", "magic", "occupancy", "page", "raisin", "rice", "room", "segment", "skin", "wilt"]
    for d, data in itertools.product(depths, datasets):
        runs.append(
            RunParams(
                method="quantbnb",
                task="classification",
                timeout=10*60,
                dataset=data,
                max_depth=d,
            )
        )
    return runs
