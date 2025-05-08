import itertools
from typing import List
from src.methods import RunParams


def setup_ablation() -> List[RunParams]:
    runs = []
    strategies = ["dfs", "bfs-lb", "bfs-curiosity", "bfs-gosdt", "dfs-prio", "and-or"]
    datasets = ["qsar", "fish", "concrete"]
    depths = [2]
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
    datasets = ["bank"]
    depths = [2]
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
