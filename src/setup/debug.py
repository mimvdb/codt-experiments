import itertools
from typing import List
from src.methods import RunParams


def setup_debug() -> List[RunParams]:
    runs = []
    methods = ["codt", "cart"]
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
                intermediates=True,
                memory_limit=3900 * 1024 * 1024 # 3900 MB
            )
        )
    datasets = ["bank"]
    for m, d, data in itertools.product(methods, depths, datasets):
        runs.append(
            RunParams(
                method=m,
                task="classification",
                timeout=120,
                dataset=data,
                max_depth=d,
                intermediates=True,
                memory_limit=3900 * 1024 * 1024 # 3900 MB
            )
        )
    return runs
