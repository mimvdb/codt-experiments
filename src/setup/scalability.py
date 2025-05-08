import itertools
from typing import List
from src.methods import RunParams


def setup_scalability() -> List[RunParams]:
    runs = []
    methods = ["codt", "quantbnb"]
    datasets = ["qsar", "fish", "concrete"]
    depths = [2]
    for m, d, data in itertools.product(methods, depths, datasets):
        runs.append(
            RunParams(
                method=m,
                task="regression",
                timeout=120,
                dataset=data,
                max_depth=d,
                strategy="dfs-prio" if m == "codt" else "",
                intermediates=True
            )
        )
    methods = ["codt", "quantbnb"]
    datasets = ["bank"]
    depths = [2]
    for m, d, data in itertools.product(methods, depths, datasets):
        runs.append(
            RunParams(
                method=m,
                task="classification",
                timeout=120,
                dataset=data,
                max_depth=d,
                strategy="dfs-prio" if m == "codt" else "",
                intermediates=True
            )
        )
    return runs
