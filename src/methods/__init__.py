from .base import BaseMethod, Run, RunOutput, RunParams

_METHODS = {}

def get_method(method_id: str) -> BaseMethod:
    if method_id not in _METHODS:
        if method_id == "cart":
            from .cart import CartRegressionMethod
            _METHODS["cart"] = CartRegressionMethod()
        elif method_id == "codt":
            from .codt import CodtMethod
            _METHODS["codt"] = CodtMethod()
        elif method_id == "quantbnb":
            # Import julia lazily because it is heavy to import.
            # TODO: Do timing for this method inside julia, so no overhead is taken in the measurements
            from .quantbnb import QuantBnBMethod
            _METHODS["quantbnb"] = QuantBnBMethod()
    
    assert method_id in _METHODS, f"Invalid method id '{method_id}' requested"
    return _METHODS[method_id]