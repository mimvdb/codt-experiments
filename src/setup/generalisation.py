import itertools
from typing import List
from src.methods import RunParams


def setup_generalisation() -> List[RunParams]:
    runs = []
    methods = ["codt", "cart"]
    datasets = ["casp", "concrete", "energy", "fish", "gas", "grid", "news", "qsar", "query1"]
    depths = [2, 3]
    test_sets = ["0", "1", "2", "3", "4"]
    for m, d, data, test_set in itertools.product(methods, depths, datasets, test_sets):
        runs.append(
            RunParams(
                method=m,
                task="regression",
                timeout=10*60,
                dataset=data,
                max_depth=d,
                test_set=test_set,
                tune=True
            )
        )
    methods = ["cart"]
    depths = [None]
    for m, d, data, test_set in itertools.product(methods, depths, datasets, test_sets):
        runs.append(
            RunParams(
                method=m,
                task="regression",
                timeout=10*60,
                dataset=data,
                max_depth=d,
                test_set=test_set,
                tune=True
            )
        )
    return runs
