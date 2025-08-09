import itertools
from typing import List
from src.data import DATASETS_REGRESSION, DATASETS_CLASSIFICATION
from src.methods import RunParams
from codt_py import all_search_strategies, all_terminal_solvers, all_branch_relaxations

TIMEOUT = 5 * 60 # timeout in seconds
main_ss = "bfs-balance-small-lb"
main_terminal_solver = "left-right"
main_branch_relaxation = "lowerbound"

def setup_ablation() -> List[RunParams]:
    return setup_ablation_ss() + setup_ablation_terminal() + setup_ablation_branch_relaxation()

def setup_ablation_ss() -> List[RunParams]:
    runs = []
    strategies = all_search_strategies()
    depths = [2, 3, 4]
    for s, d, data in itertools.product(strategies, depths, DATASETS_REGRESSION):
        runs.append(
            RunParams(
                method="codt",
                task="regression",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                strategy=s,
                intermediates=True
            )
        )
    for s, d, data in itertools.product(strategies, depths, DATASETS_CLASSIFICATION):
        runs.append(
            RunParams(
                method="codt",
                task="classification",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                strategy=s,
                intermediates=True
            )
        )
    return runs

def setup_ablation_terminal() -> List[RunParams]:
    runs = []
    terminal_solvers = all_terminal_solvers()
    terminal_solvers.remove(main_terminal_solver)  # Exclude the main terminal solver, results are already in albation_ss.
    terminal_solvers.remove("d2")  # Exclude d2: not implemented yet.
    depths = [2, 3, 4]
    for t, d, data in itertools.product(terminal_solvers, depths, DATASETS_REGRESSION):
        runs.append(
            RunParams(
                method="codt",
                task="regression",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                terminal_solver=t,
                intermediates=True
            )
        )
    for t, d, data in itertools.product(terminal_solvers, depths, DATASETS_CLASSIFICATION):
        runs.append(
            RunParams(
                method="codt",
                task="classification",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                terminal_solver=t,
                intermediates=True
            )
        )
    return runs

def setup_ablation_branch_relaxation() -> List[RunParams]:
    runs = []
    relaxations = all_branch_relaxations()
    relaxations.remove(main_branch_relaxation)  # Exclude the main branch relaxation, results are already in ablation_ss.
    relaxations.remove("exact")  # Exclude "exact": not implemented yet.
    depths = [2, 3, 4]
    for r, d, data in itertools.product(relaxations, depths, DATASETS_REGRESSION):
        runs.append(
            RunParams(
                method="codt",
                task="regression",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                branch_relaxation=r,
                intermediates=True
            )
        )
    for r, d, data in itertools.product(relaxations, depths, DATASETS_CLASSIFICATION):
        runs.append(
            RunParams(
                method="codt",
                task="classification",
                timeout=TIMEOUT,
                dataset=data,
                max_depth=d,
                branch_relaxation=r,
                intermediates=True
            )
        )
    return runs