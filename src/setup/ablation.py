import itertools
from typing import List
from src.methods import RunParams
from codt_py import all_search_strategies

def setup_ablation() -> List[RunParams]:
    runs = []
    strategies = all_search_strategies()
    datasets = ["casp", "concrete", "energy", "fish", "gas", "grid", "news", "qsar", "query1"]
    depths = [2, 3, 4]
    for s, d, data in itertools.product(strategies, depths, datasets):
        runs.append(
            RunParams(
                method="codt",
                task="regression",
                timeout=15*60,
                dataset=data,
                max_depth=d,
                strategy=s,
                intermediates=True
            )
        )
    datasets = ["avila", "bank", "bean", "bidding", "eeg", "fault", "htru", "magic", "occupancy", "page", "raisin", "rice", "room", "segment", "skin", "wilt"]
    for s, d, data in itertools.product(strategies, depths, datasets):
        runs.append(
            RunParams(
                method="codt",
                task="classification",
                timeout=15*60,
                dataset=data,
                max_depth=d,
                strategy=s,
                intermediates=True
            )
        )
    return runs
