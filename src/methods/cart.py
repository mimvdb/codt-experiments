import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from .base import BaseMethod, RunOutput, RunParams

class CartRegressionMethod(BaseMethod):
    def __init__(self):
        super().__init__("cart", ["regression"])

    def run_method(self, X_train, y_train, X_test, y_test, params: RunParams):
        total_train_var = np.std(y_train) * np.std(y_train)

        start_time = time.time() # Start timer after reading data

        if params.tune:
            model = DecisionTreeRegressor()
            parameters = {
                "max_depth": [params.max_depth],
                "ccp_alpha": np.array([0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001]) * total_train_var
            }

            tuning_model = GridSearchCV(
                model, param_grid=parameters, scoring="neg_mean_squared_error", cv=5, verbose=0
            )
            tuning_model.fit(X_train, y_train)
            model = DecisionTreeRegressor(**tuning_model.best_params_)
            tuning_output = tuning_model.cv_results_
        else:
            model = DecisionTreeRegressor(max_depth=params.max_depth, ccp_alpha=params.cp * total_train_var)
            tuning_output = None

        model.fit(X_train, y_train)

        duration = time.time() - start_time            

        return RunOutput(
            time=duration,
            train_score=r2_score(y_train, model.predict(X_train)),
            test_score=r2_score(y_test, model.predict(X_test)),
            depth=model.get_depth(),
            leaves=model.get_n_leaves(),
            output="",
            intermediates=None,
            tuning_output=tuning_output)
