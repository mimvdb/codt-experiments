import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    check_is_fitted,
    validate_data,
)
from .base import BaseMethod, RunParams

# os.environ["PYTHON_JULIACALL_SYSIMAGE"] = "sys_precompiled.so"
from juliacall import Main as jl

jl.seval('using QuantBnBWrapper')


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
            _, self.tree_, self.time_ = jl.optimal_classification_2d(X, y_quant)
        elif self.max_depth == 3:
            _, self.tree_, self.time_ = jl.optimal_classification_3d(X, y_quant, self.timeout)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.take(
            self.classes_,
            np.argmax(jl.tree_eval(self.tree_, X, 2, len(self.classes_)), axis=1),
        )


class QuantBnBDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth, timeout):
        self.max_depth = max_depth
        self.timeout = timeout

    def fit(self, X, y):
        X, y = validate_data(
            self, X, y, ensure_min_samples=2, dtype=np.float64, y_numeric=True
        )

        # Quant-BnB wants a column vector
        y_quant = np.array([y]).T

        # Quant-BnB can only do depth 2 or 3 trees.
        assert self.max_depth in [2, 3]
        if self.max_depth == 2:
            _, self.tree_, self.time_ = jl.optimal_regression_2d(X, y_quant)
        elif self.max_depth == 3:
            _, self.tree_, self.time_ = jl.optimal_regression_3d(X, y_quant, self.timeout)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return jl.tree_eval(self.tree_, X, 2, 1)


class QuantBnBMethod(BaseMethod):
    def __init__(self, task):
        super().__init__("quantbnb", task)
        if task == "classification":
            self.model = QuantBnBDecisionTreeClassifier
        elif task == "regression":
            self.model = QuantBnBDecisionTreeRegressor

    def tree_to_list(self, tree):
        if len(tree) == 4:
            return [
                tree[0] - 1, # 1-indexed to 0-indexed features
                tree[1],
                self.tree_to_list(tree[2]),
                self.tree_to_list(tree[3]),
            ]
        else:
            if self.task == "classification":
                return np.argmax(tree)
            elif self.task == "regression":
                return np.array(tree)[0, 0]

    def train_model(self, X, y, params: RunParams):
        model = self.model(max_depth=params.max_depth, timeout=params.timeout)
        model.fit(X, y)

        return (model, {"tree": self.tree_to_list(model.tree_), "time": model.time_})
