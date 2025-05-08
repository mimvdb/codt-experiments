from typing import List
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .base import BaseMethod, RunParams


def desired_alphas(alpha_thresholds):
    """Returns the geometric mean between each threshold. This makes the alpha more robust against slight dataset changes."""
    alpha_thresholds.append(1.0)
    alphas = []
    for i in range(len(alpha_thresholds) - 1):
        alphas.append(np.sqrt(alpha_thresholds[i] * alpha_thresholds[i + 1]))
    return alphas


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
                value = tree.value[node_id][0]
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
            ccp_path = model.cost_complexity_pruning_path(X, y)
            ccp_alphas = desired_alphas(ccp_path.ccp_alphas)
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
