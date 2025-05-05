import numpy as np
import time
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted, validate_data
from sklearn.metrics import r2_score
from .base import BaseMethod, RunOutput, RunParams

from juliacall import Main as jl
jl.seval("include(\"Quant-BnB/call.jl\")")
jl.seval("include(\"Quant-BnB/gen_data.jl\")") # for tree_eval

class QuantBnBDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64)
        self.classes_, y = np.unique(y, return_inverse=True)

        # Quant-BnB expects a one-hot encoded Y array.
        y_quant = np.zeros((y.size, y.max() + 1))
        y_quant[np.arange(y.size), y] = 1

        # Quant-BnB can only do depth 2 or 3 trees.
        assert self.max_depth == 2

        _, self.tree_ = jl.optimal_classification_2d(X, y_quant)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.take(self.classes_, jl.tree_eval(self.tree_, X, 2, len(self.classes_)))

class QuantBnBDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, y):
        X, y = validate_data(self, X, y, ensure_min_samples=2, dtype=np.float64, y_numeric=True)

        # Quant-BnB wants a column vector
        y_quant = np.array([y]).T

        # Quant-BnB can only do depth 2 or 3 trees.
        assert self.max_depth == 2

        _, self.tree_ = jl.optimal_regression_2d(X, y_quant)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return jl.tree_eval(self.tree_, X, 2, 1)


class QuantBnBMethod(BaseMethod):
    def __init__(self):
        super().__init__("quantbnb", ["regression"])

    def run_method(self, X_train, y_train, X_test, y_test, params: RunParams):
        start_time = time.time() # Start timer after reading data

        model = QuantBnBDecisionTreeRegressor(max_depth=params.max_depth)
        tuning_output = None

        model.fit(X_train, y_train)

        duration = time.time() - start_time            

        return RunOutput(
            time=duration,
            train_score=r2_score(y_train, model.predict(X_train)),
            test_score=r2_score(y_test, model.predict(X_test)),
            depth=0, # TODO model.get_depth(),
            leaves=0, # TODO model.get_n_leaves(),
            output="",
            intermediates=None,
            tuning_output=tuning_output)
