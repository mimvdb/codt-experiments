import numpy as np
import time
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted, validate_data
from sklearn.metrics import accuracy_score, r2_score
from .base import BaseMethod, RunOutput, RunParams

from juliacall import Main as jl
jl.seval("include(\"Quant-BnB/call.jl\")")
jl.seval("include(\"Quant-BnB/gen_data.jl\")") # for tree_eval

class QuantBnBDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth, timeout):
        self.max_depth = max_depth
        self.timeout = timeout

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)

        # Quant-BnB expects a one-hot encoded Y array.
        y_quant = np.zeros((y.size, y.max() + 1))
        y_quant[np.arange(y.size), y] = 1

        # Quant-BnB can only do depth 2 or 3 trees.
        assert self.max_depth in [2, 3]
        if self.max_depth == 2:
            _, self.tree_ = jl.optimal_classification_2d(X, y_quant)
        elif self.max_depth == 3:
            _, self.tree_ = jl.optimal_classification_3d(X, y_quant, 60) # TODO timelimit

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.take(self.classes_, jl.tree_eval(self.tree_, X, 2, len(self.classes_)))

class QuantBnBDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth, timeout):
        self.max_depth = max_depth
        self.timeout = timeout

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64, y_numeric=True)

        # Quant-BnB wants a column vector
        y_quant = np.array([y]).T

        # Quant-BnB can only do depth 2 or 3 trees.
        assert self.max_depth in [2, 3]
        if self.max_depth == 2:
            _, self.tree_ = jl.optimal_regression_2d(X, y_quant)
        elif self.max_depth == 3:
            _, self.tree_ = jl.optimal_regression_3d(X, y_quant, 60) # TODO timelimit


    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return jl.tree_eval(self.tree_, X, 2, 1)


class QuantBnBMethod(BaseMethod):
    def __init__(self, task):
        super().__init__("quantbnb", task)
        if task == "classification":
            self.model = QuantBnBDecisionTreeClassifier
            self.score_func = accuracy_score
        elif task == "regression":
            self.model = QuantBnBDecisionTreeRegressor
            self.score_func = r2_score

    def run_method(self, X_train, y_train, X_test, y_test, params: RunParams):
        start_time = time.time() # Start timer after reading data

        model = self.model(max_depth=params.max_depth, timeout=params.timeout)

        model.fit(X_train, y_train)

        # TODO: Do timing for this method inside julia, so no overhead is taken in the measurements
        duration = time.time() - start_time            

        return RunOutput(
            time=duration,
            train_score=self.score_func(y_train, model.predict(X_train)),
            test_score=self.score_func(y_test, model.predict(X_test)),
            depth=0, # TODO model.get_depth(),
            leaves=0, # TODO model.get_n_leaves(),
            output="")
