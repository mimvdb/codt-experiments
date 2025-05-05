import itertools
from typing import List
from src.methods import RunParams

def setup_debug() -> List[RunParams]:
    runs = []
    methods = ["codt", "cart", "quantbnb"]
    datasets = ["qsar", "fish", "concrete"]
    depths = [2]
    for m, d, data in itertools.product(methods, depths, datasets):
        runs.append(RunParams(m, "regression", 120, data, data, d))
    return runs
