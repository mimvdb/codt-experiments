import numpy as np
import time
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, asdict
from io import StringIO
from typing import Dict, List, Optional, Tuple
from traceback import format_exc
from sklearn.base import BaseEstimator
from src.data import get_dataset, get_test_indices


def depth_from_tree(tree):
    if tree is None:
        return -1
    elif hasattr(tree, "__len__") and len(tree) == 4:
        return max(depth_from_tree(tree[2]), depth_from_tree(tree[3])) + 1
    else:
        return 0


def leaves_from_tree(tree):
    if tree is None:
        return -1
    elif hasattr(tree, "__len__") and len(tree) == 4:
        return leaves_from_tree(tree[2]) + leaves_from_tree(tree[3])
    else:
        return 1


@dataclass(frozen=True)
class RunParams():
    """ All parameters for a single run of a method.

    Attributes:
        method: String id of the method
        task: Task of decision tree. E.g. "classification" or "regression"
        timeout: Timeout in seconds.
        dataset: Name of the dataset.
        max_depth: Enforce a maximum depth limit for the tree.
        test_set: Index set to use as test set, empty string for none.
        cp: Complexity penalty.
        strategy: Use the specified search strategy
        upperbound: Use the specified upperbounding strategy
        terminal_solver: Use the specified terminal solver. I.e. "leaf" or "d1.5" or "d2"
        intermediates: If true, keep scores of intermediate solutions.
        tune: If true, tune hyperparameters for out-of-sample performance.
        branch_relaxation: If enabled, use branch relaxation in the solver.
        memory_limit: Optional memory limit in bytes.
    """
    method: str
    task: str
    timeout: int
    dataset: str
    max_depth: int
    test_set: str = ""
    cp: float = 0.0
    strategy: str = "bfs-balance-small-lb"
    upperbound: str = "for-remaining-interval"
    terminal_solver: str = "left-right"
    intermediates: bool = False
    tune: bool = False
    branch_relaxation: str = "lowerbound"
    memory_limit: Optional[int] = 4000 * 1024 * 1024  # 4GB

    def as_dict(self):
        return asdict(self)

@dataclass
class RunOutput():
    """ The result of running a method for a certain set of parameters.

    Attributes:
        time: The time in seconds that it took to train the model.
        train_score: The accuracy or r2 score of the resulting tree on the train dataset.
        test_score: The accuracy or r2 score of the resulting tree on the test dataset.
        depth: The deepest path in the resulting tree, a tree with only a root node is depth 0.
        leaves: The number of leaf nodes in the resulting tree.
        output: Output string, used for debugging.
        tree: The tree, format is a nested list of [feature, threshold, left_child, right_child] for internal nodes and the label for leafs
        intermediate_lbs: Optionally, a list of tuples with the training scores, graph expansions, and time of intermediate solutions.
        intermediate_ubs: Optionally, a list of tuples with the lower bounds, graph expansions, and time of intermediate solutions.
        memory_usage_bytes: Optionally, the memory in bytes used during training.
        expansions: Optionally, the total number of graph expansions done during training.
        tuning_output: Optionally, the extra information during tuning.
    """
    time: float
    train_score: float
    test_score: float
    depth: int
    leaves: int
    output: str
    tree: Optional[List | float]
    intermediate_lbs: Optional[List[Tuple[float, int, float]]]
    intermediate_ubs: Optional[List[Tuple[float, int, float]]]
    memory_usage_bytes: Optional[int]
    expansions: Optional[int]
    tuning_output: Optional[Dict]

    @staticmethod
    def empty_with_output(output: str):
        return RunOutput(-1.0, 0.0, 0.0, 0, 0, output, None, None, None, None, None, None)


@dataclass
class Run():
    p: RunParams
    o: RunOutput

    def as_dict(self):
        return {"p": self.p.as_dict(), "o": asdict(self.o)}

class BaseMethod(ABC):
    def __init__(self, method_id: str, task: str):
        self.method_id = method_id
        self.task = task

    def run(self, params: RunParams, scorer = None) -> RunOutput:
        assert params.method == self.method_id, f"{params.method} == {self.method_id}"
        assert params.task in self.task, f"{params.task} == {self.task}"
        assert params.task == "regression" or not params.tune # OOS experiments only for regression

        X, y = get_dataset(params.dataset, params.task)
        if params.test_set == "":
            X_train, y_train, X_test, y_test = X, y, None, None
        else:
            test_idx = get_test_indices(params.dataset, params.task, params.test_set)
            train_select = np.full(y.shape, True)
            train_select[test_idx] = False
            X_train, y_train, X_test, y_test = X[train_select], y[train_select], X[test_idx], y[test_idx]

        stdout_io = StringIO()
        stderr_io = StringIO()

        def append_std(output):
            stdout_str = stdout_io.getvalue()
            stderr_str = stderr_io.getvalue()
            if len(stdout_str) > 0: output += f"\nSTDOUT\n{stdout_str}"
            if len(stderr_str) > 0: output += f"\nSTDERR\n{stderr_str}"
            return output

        try:
            with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
                start_time = time.time()
                (model, extra) = self.train_model(X_train, y_train, params)
                duration = time.time() - start_time
                
                tree = extra.get("tree")

                if scorer is not None:
                    score_func = lambda X, y: scorer(y, model.predict(X))
                else:
                    score_func = model.score

                train_score=score_func(X_train, y_train)
                test_score=0.0 if X_test is None else score_func(X_test, y_test)

                result = RunOutput(
                    time=extra.get("time", duration),
                    train_score=train_score,
                    test_score=test_score,
                    depth=depth_from_tree(tree),
                    leaves=leaves_from_tree(tree),
                    output="",
                    tree=tree,
                    intermediate_lbs=extra.get("intermediate_lbs") if params.intermediates else None,
                    intermediate_ubs=extra.get("intermediate_ubs") if params.intermediates else None,
                    memory_usage_bytes=extra.get("memory_usage_bytes"),
                    expansions=extra.get("expansions"),
                    tuning_output=extra.get("tuning_output"))

                result.output = append_std(result.output)
        except Exception:
            result = RunOutput.empty_with_output(format_exc())
            result.output = append_std(result.output)
        
        return result


    @abstractmethod
    def train_model(self, X, y, params: RunParams) -> Tuple[BaseEstimator, Dict]:
        """ Use the method to train a model
        """
