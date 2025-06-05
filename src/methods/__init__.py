from .base import BaseMethod, Run, RunOutput, RunParams

def get_method(method_id: str, task: str) -> BaseMethod:
    if task == "classification":
        return get_classification_method(method_id)
    elif task == "regression":
        return get_regression_method(method_id)
    else:
        assert False, "Only classification and regression is supported"

def get_classification_method(method_id: str) -> BaseMethod:
    if method_id == "cart":
        from .cart import CartMethod
        return CartMethod("classification")
    elif method_id == "codt":
        from .codt import CodtMethod
        return CodtMethod("classification")
    elif method_id == "quantbnb":
        # Import lazily because julia is heavy to import.
        from .quantbnb import QuantBnBMethod
        return QuantBnBMethod("classification")
    else:
        assert False, f"Invalid classification method id '{method_id}' requested"

def get_regression_method(method_id: str) -> BaseMethod:
    if method_id == "cart":
        from .cart import CartMethod
        return CartMethod("regression")
    elif method_id == "codt":
        from .codt import CodtMethod
        return CodtMethod("regression")
    elif method_id == "quantbnb":
        # Import lazily because julia is heavy to import.
        from .quantbnb import QuantBnBMethod
        return QuantBnBMethod("regression")
    else:
        assert False, f"Invalid regression method id '{method_id}' requested"
