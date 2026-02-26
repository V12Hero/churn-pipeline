# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.

import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


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
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def load_object(model_params: dict, verbose: bool = False) -> Any:
    """Load object.

    Loads a object from the class given as a parameter.

    Args:
        model_params: dictionary of parameters for train
        verbose: print logs.

    Returns:
        Any python object.
    """
    model_class = model_params["class"]
    model_kwargs = model_params["kwargs"]
    if model_params["kwargs"] is None:
        model_kwargs = {}
    if verbose:
        logger.info(f"loading with {model_params}")
    python_object = _load_obj(model_class)(**model_kwargs)
    return python_object


def load_object_with_arg(model_params: dict, extra_args: Dict):
    """Load from catalog sklearn transformer.

    Loads a regressor object based on given parameters.

    Args:
        model_params: dictionary of parameters for train

    Returns:
        sklearn compatible model
    """
    model_class = model_params["class"]
    model_kwargs = model_params["kwargs"]
    extra_args.update(model_kwargs)
    if model_params["kwargs"] is None:
        model_kwargs = {}
    baseline_model = _load_obj(model_class)(**extra_args)
    return baseline_model
