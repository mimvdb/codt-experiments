import numpy as np
from sklearn.model_selection import GridSearchCV
from codt_py import (
    OptimalDecisionTreeClassifier,
    OptimalDecisionTreeRegressor,
)
from .base import BaseMethod, RunParams


class CodtMethod(BaseMethod):
    def __init__(self, task):
        super().__init__("codt", task)
        if task == "classification":
            self.model = OptimalDecisionTreeClassifier
        elif task == "regression":
            self.model = OptimalDecisionTreeRegressor

    def train_model(self, X, y, params: RunParams):
        if params.tune:
            time_per_cp = int(params.timeout / 11) # 10 CP + 1 Final
            time_per_tune = int(time_per_cp / 5) # 5 fold cross validation per parameter.
            model = OptimalDecisionTreeRegressor()
            parameters = {
                "max_depth": [params.max_depth],
                "complexity_cost": np.array(
                    [
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
                ),
                "timeout": [time_per_tune],
                "memory_limit": [params.memory_limit],
            }

            tuning_model = GridSearchCV(
                model,
                param_grid=parameters,
                scoring="neg_mean_squared_error",
                cv=5,
                verbose=0,
            )
            tuning_model.fit(X, y)
            tuning_model.best_params_["timeout"] = time_per_cp  # Use larger timeout for the final model.
            model = OptimalDecisionTreeRegressor(**tuning_model.best_params_)
            tuning_output = tuning_model.cv_results_
        else:
            model = self.model(
                max_depth=params.max_depth,
                strategy=params.strategy,
                complexity_cost=params.cp,
                timeout=params.timeout,
                upperbound=params.upperbound,
                terminal_solver=params.terminal_solver,
                intermediates=params.intermediates,
                branch_relaxation=params.branch_relaxation,
                memory_limit=params.memory_limit,
            )
            tuning_output = None

        model.fit(X, y)
        return (
            model,
            {
                "tuning_output": tuning_output,
                "tree": model.get_tree(),
                "intermediate_lbs": model.intermediate_lbs(),
                "intermediate_ubs": model.intermediate_ubs(),
                "memory_usage_bytes": model.memory_usage_bytes(),
                "expansions": model.expansions(),
            },
        )
