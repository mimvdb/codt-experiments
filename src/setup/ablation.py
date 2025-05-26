import itertools
from typing import List
from src.methods import RunParams
from codt_py import all_search_strategies

def setup_ablation() -> List[RunParams]:
    runs = []
    strategies = all_search_strategies()
    datasets = ["concrete", "fish", "qsar", "query1"]
    depths = [2, 3]
    for s, d, data in itertools.product(strategies, depths, datasets):
        runs.append(
            RunParams(
                method="codt",
                task="regression",
                timeout=120,
                dataset=data,
                max_depth=d,
                strategy=s,
                intermediates=True
            )
        )
    datasets = ["bank", "bidding", "eeg", "fault", "page", "raisin", "room"]
    for s, d, data in itertools.product(strategies, depths, datasets):
        runs.append(
            RunParams(
                method="codt",
                task="classification",
                timeout=120,
                dataset=data,
                max_depth=d,
                strategy=s,
                intermediates=True
            )
        )
    return runs
