from .base import BaseMethod, Run, RunOutput, RunParams

_CLASSIFICATION_METHODS = {}
_REGRESSION_METHODS = {}

def get_classification_method(method_id: str) -> BaseMethod:
    if method_id not in _CLASSIFICATION_METHODS:
        if method_id == "cart":
            from .cart import CartMethod
            _REGRESSION_METHODS["cart"] = CartMethod("classification")
        elif method_id == "codt":
            from .codt import CodtMethod
            _CLASSIFICATION_METHODS["codt"] = CodtMethod("classification")
        elif method_id == "quantbnb":
            # Import lazily because julia is heavy to import.
            from .quantbnb import QuantBnBMethod
            _CLASSIFICATION_METHODS["quantbnb"] = QuantBnBMethod("classification")
    
    assert method_id in _CLASSIFICATION_METHODS, f"Invalid method id '{method_id}' requested"
    return _CLASSIFICATION_METHODS[method_id]

def get_regression_method(method_id: str) -> BaseMethod:
    if method_id not in _REGRESSION_METHODS:
        if method_id == "cart":
            from .cart import CartMethod
            _REGRESSION_METHODS["cart"] = CartMethod("regression")
        elif method_id == "codt":
            from .codt import CodtMethod
            _REGRESSION_METHODS["codt"] = CodtMethod("regression")
        elif method_id == "quantbnb":
            # Import lazily because julia is heavy to import.
            from .quantbnb import QuantBnBMethod
            _REGRESSION_METHODS["quantbnb"] = QuantBnBMethod("regression")
    
    assert method_id in _REGRESSION_METHODS, f"Invalid method id '{method_id}' requested"
    return _REGRESSION_METHODS[method_id]


