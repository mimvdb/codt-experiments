from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, asdict
from io import StringIO
from typing import Dict, List, Optional
from traceback import format_exc
from src.util import get_test_set, get_train_set

@dataclass
class RunParams():
    """ All parameters for a single run of a method.

    Attributes:
        method: String id of the method
        task: Task of decision tree. E.g. "classification" or "regression"
        timeout: Timeout in seconds.
        train_set: Name of the train dataset.
        test_set: Name of the test dataset.
        max_depth: Enforce a maximum depth limit for the tree.
        cp: Complexity penalty.
        strategy: Use the specified search strategy
        intermediates: If true, keep scores of intermediate solutions.
        tune: If true, tune hyperparameters for out-of-sample performance.
        base_case_solver: The base case of a decision tree. I.e. "leaf" or "d1.5" or "d2"
    """
    method: str
    task: str
    timeout: int
    train_set: str
    test_set: str
    max_depth: int
    cp: float = 0.0
    strategy: str = ""
    intermediates: bool = False
    tune: bool = False

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
        intermediates: Optionally, the training scores of intermediate solutions.
        tuning_output: Optionally, the extra information during tuning.
    """
    time: float
    train_score: float
    test_score: float
    depth: int
    leaves: int
    output: str
    intermediates: Optional[List[float]] = None
    tuning_output: Optional[Dict] = None

    @staticmethod
    def empty_with_output(output: str):
        return RunOutput(-1.0, 0.0, 0.0, 0, 0, output)


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

    def run(self, params: RunParams) -> RunOutput:
        assert params.method == self.method_id
        assert params.task in self.task
        assert params.task == "regression" or not params.tune # OOS experiments only for regression

        X_train, y_train = get_train_set(params.train_set, params.task)
        X_test, y_test = get_test_set(params.test_set, params.task)

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
                result = self.run_method(X_train, y_train, X_test, y_test, params)
                result.output = append_std(result.output)
        except Exception as e:
            result = RunOutput.empty_with_output(format_exc(e))
            result.output = append_std(result.output)
        
        return result


    @abstractmethod
    def run_method(self, X_train, y_train, X_test, y_test, params: RunParams) -> RunOutput:
        """ Run the method and produce a result
        """
