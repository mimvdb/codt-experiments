from typing import List
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .base import BaseMethod, RunParams


def desired_alphas(alpha_thresholds):
    """Returns the geometric mean between each threshold. This makes the alpha more robust against slight dataset changes."""
    np.append(alpha_thresholds, 1.0)
    alphas = []
    for i in range(len(alpha_thresholds) - 1):
        # max() for rounding errors to below 0
        a1 = max(0, alpha_thresholds[i])
        a2 = max(0, alpha_thresholds[i + 1])
        # Skip if equal due to rounding errors
        if a1 == a2: continue
        alphas.append(np.sqrt(a1 * a2))
    
    # To prevent very long runtimes in CART, limit to 30 parameters
    max_alphas = 30
    if len(alphas) <= max_alphas: return alphas
    limit_set = []
    float_i = 0.0
    float_step = len(alphas) / max_alphas

    while len(limit_set) < 30:
        limit_set.append(alphas[np.floor(float_i).astype(int)])
        float_i += float_step

    return limit_set


def complexity_path_alphas(model, X, y):
    ccp_path = model.cost_complexity_pruning_path(X, y)
    return desired_alphas(ccp_path.ccp_alphas)


def constant_alphas():
    return [
        0.1,
        0.05,
        0.025,
        0.01,
        0.0075,
        0.005,
        0.0025,
        0.001,
        0.0005,
        0.0001,
    ]


class CartMethod(BaseMethod):
    def __init__(self, task):
        super().__init__("cart", task)
        if task == "classification":
            self.model = DecisionTreeClassifier
        elif task == "regression":
            self.model = DecisionTreeRegressor

    def tree_to_list(self, tree, node_id) -> List | float:
        # TREE_UNDEFINED == -2, means its a leaf
        if tree.feature[node_id] == -2:
            if tree.n_outputs == 1:
                value = tree.value[node_id][0][0]
            else:
                value = tree.value[node_id].T[0]
            return np.argmax(value) if self.task == "classification" else value
        else:
            # Children exist
            return [
                tree.feature[node_id],
                tree.threshold[node_id],
                self.tree_to_list(tree, tree.children_left[node_id]),
                self.tree_to_list(tree, tree.children_right[node_id]),
            ]

    def train_model(self, X, y, params: RunParams):
        x = np.std(y) * np.std(y) if self.task == "regression" else len(y)

        if params.tune:
            model = DecisionTreeRegressor(max_depth=params.max_depth)
            ccp_alphas = complexity_path_alphas(model, X, y)
            # ccp_alphas = constant_alphas()
            parameters = {"max_depth": [params.max_depth], "ccp_alpha": ccp_alphas}

            tuning_model = GridSearchCV(
                model,
                param_grid=parameters,
                scoring="neg_mean_squared_error",
                cv=5,
                verbose=0,
            )
            tuning_model.fit(X, y)
            model = DecisionTreeRegressor(**tuning_model.best_params_)
            tuning_output = tuning_model.cv_results_
        else:
            model = self.model(max_depth=params.max_depth, ccp_alpha=params.cp * x)
            tuning_output = None

        model.fit(X, y)
        extra = {
            "tuning_output": tuning_output,
            "tree": self.tree_to_list(model.tree_, 0),
        }
        return (model, extra)
