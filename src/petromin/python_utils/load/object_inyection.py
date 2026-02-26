import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

OBJECT_KW = "object"
INSTANTIATE_KW = "instantiate"


def _load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def load_object(object_params: dict) -> Any:
    """Load object.

    Loads a object from the class given as a parameter.

    Args:
        object_params: dictionary of parameters for object injection

    Returns:
        Any python object.
    """
    model_class = object_params["class"]
    model_kwargs = object_params["kwargs"]

    if object_params["kwargs"] is None:
        model_kwargs = {}

    python_object = _load_obj(model_class)(**model_kwargs)
    return python_object


def load_estimator(object_params: dict) -> Any:
    """Load estimator.

    Loads an estimator object based on given parameters.
    Checks that loaded object has ``.fit`` and ``.predict`` methods.

    Used to load estimator objects in a Kedro pipeline.

    Args:
        object_params (dict): it must have class and kwargs
        parameters

    Raises:
        AttributeError: if returned estimator doesn't
         have a ``.fit`` and ``.predict`` method.

    Returns:
        sklearn compatible estimator
    """
    model_class = object_params["class"]
    model_kwargs = object_params["kwargs"]

    if object_params["kwargs"] is None:
        model_kwargs = {}

    estimator = _load_obj(model_class)(**model_kwargs)
    if getattr(estimator, "fit", None) is None:
        raise AttributeError("Model object must have a .fit method")

    if getattr(estimator, "predict", None) is None:
        raise AttributeError("Model object must have a .predict method.")

    return estimator
