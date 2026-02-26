"""Hypertune cross validation."""
import logging
import typing as tp
from copy import deepcopy

import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from petromin.modelling.evaluate.metrics import transform_output_dict
from petromin.python_utils.load.object_inyection import load_object
from petromin.python_utils.typing.tensors import Matrix, Tensor

from .inyect_trial import inject_trial_parameter

logger = logging.getLogger(__name__)


def save_best_model(study: optuna.Study, trial: optuna.trial):
    """Callback to save best model params."""
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="estimator_params", value=trial.user_attrs["estimator_params"])


def hypertune_cross_validation_objective_function(
    trial: optuna.Trial,
    **kwargs,
) -> float:
    """Hypertune thorough cross validation a Model pipeline.

    It takes a trial object, and a kwargs of parameters, and returns the
    cross validation result for a given trial model pipeline.

    Args:
      trial (optuna.Trial): optuna.Trial
      **kwargs (tp.Any): kwargs of parameters for cross validation.

    Returns:
      Cross validation error.
    """
    params = kwargs.get("params", {})
    X = kwargs.get("X")
    y = kwargs.get("y")

    # inject trial parameters
    new_data_params = inject_trial_parameter(deepcopy(params), trial)
    trial.set_user_attr(key="estimator_params", value=new_data_params)

    # build meta estimator
    estimator = load_object(new_data_params["model_artifact"]["model"])

    # cross validation score
    score = cross_validate_estimator(
        estimator,
        X,
        y,
        deepcopy(new_data_params),
    )

    return score


def hypertune_cross_validated_model(
    X: pd.DataFrame, y: pd.DataFrame, params: dict
) -> dict[optuna.Study, dict]:
    """Performs hyperparameter tuning using cross-validation and Optuna library in Python.

    Args:
      X (pd.DataFrame): A pandas DataFrame containing the input features for the model.
      y (pd.DataFrame): Dataframe indexed with the target variable or dependent variable
        in a machine learning model.
      params (tp.Dict): The `params` parameter is a dictionary containing various
        hyperparameters and settings for the function `hypertune_cross_validated_model()`.

    Returns:
      a dictionary containing the optuna study object and the best trial parameters for
    the estimator.
    """
    kwargs_study = params.get("optuna")["kwargs_study"]
    kwargs_optimize = params.get("optuna")["kwargs_optimize"]
    sampler = load_object(params.get("optuna")["sampler"])
    pruner = load_object(params.get("optuna")["pruner"])

    # kwargs to inject in the objective function
    kwargs = {
        "X": deepcopy(X),
        "y": deepcopy(y),
        "params": deepcopy(params),
    }

    # optuna objective function
    objective = lambda trial: hypertune_cross_validation_objective_function(  # noqa: E731
        trial, **deepcopy(kwargs)
    )

    # optimize study
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        **kwargs_study,
    )

    study.optimize(objective, **kwargs_optimize)

    # retrieve the information of the best model
    best_trial = study.best_trial
    best_estimator_number = best_trial.number
    best_estimator_params = study.trials[best_estimator_number].user_attrs["estimator_params"]
    best_value = hypertune_cross_validation_objective_function(
        best_trial, callbacks=[save_best_model], **deepcopy(kwargs)
    )
    logger.info(f"best_value: {best_value}")

    # compute test scoring
    cross_validation_metrics = _compute_test_cross_validation_metrics(
        X, y, best_estimator_params, best_value
    )
    # mlflow logging metric transform
    cross_validation_metrics = transform_output_dict(cross_validation_metrics)

    return {
        "study": study,
        "best_trial_params": best_estimator_params,
        "cross_validation_metrics": cross_validation_metrics,
    }


def _compute_test_cross_validation_metrics(
    X: pd.DataFrame,
    y: pd.DataFrame,
    best_estimator_params: tp.Dict[str, tp.Any],
    best_value: float,
    scoring_metrics: tp.List[str] = [
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "r2",
        "neg_mean_absolute_percentage_error",
        "explained_variance",
        "max_error",
    ],
) -> tp.Dict[str, float]:
    """
    The function computes cross-validation metrics
    for a given dataset using the best estimator parameters.

    Args:
      X (pd.DataFrame): X is a pandas DataFrame containing the input features for the model.
      y (pd.DataFrame): The parameter `y` is a pandas DataFrame that represents the target variable or
    the dependent variable in your dataset. It contains the values that you are trying to predict or
    model.
      best_estimator_params (tp.Dict[str, tp.Any]): The `best_estimator_params` parameter is a
    dictionary that contains the parameters of the best estimator model. It typically includes
    information such as the model artifact (serialized model object), cross-validation score, and other
    relevant parameters.
      best_value: The parameter `best_value` is the best value obtained for the specified metric during
    the cross-validation process. It is used to store the best value for the metric in the returned
    dictionary.

    Returns:
      a dictionary containing the best value for the specified metric, as well as the scores for various
    other metrics obtained through cross-validation.
    """
    metric = best_estimator_params["cv_score"]["kwargs"]["scoring"]
    best_value_metric = {"best_value_" + metric: best_value}
    test_params = deepcopy(best_estimator_params)
    for metric in scoring_metrics:
        estimator = load_object(best_estimator_params["model_artifact"]["model"])
        test_params["cv_score"]["kwargs"]["scoring"] = metric
        score = cross_validate_estimator(
            estimator,
            X,
            y,
            deepcopy(test_params),
        )
        best_value_metric[metric] = score
    return best_value_metric


def cross_validate_estimator(
    estimator: Pipeline,
    X: tp.Union[Matrix, Tensor],
    y: tp.Union[Matrix, Tensor],
    params: dict,
) -> float:
    """Perform cross-validation a estimator using the input data and parameters.

    Args:
    estimator (Pipeline): A pipeline containing the stacked models and the
        final meta-model.
    X (Union[Matrix, Tensor]): Input data matrix or tensor.
    y (Union[Matrix, Tensor]): Target data matrix or tensor.
    params (Dict): A dictionary containing the following keys:
        - "cv_strategy" (str): The type of cross-validation
            strategy to be used.
        - "cv_score" (Dict): A dictionary containing
        the following keys:
        - "type" (str): The type of cross-validation score to be used.
        - "kwargs" (Dict): A dictionary containing the arguments to be
            passed to the score function.

    Returns:
        float: The mean score of the cross-validation.
    """
    # cross validate the model
    cv_strategy = load_object(params["cv_strategy"])
    cv_score_params = params["cv_score"]
    cv_score_params["kwargs"]["X"] = X
    cv_score_params["kwargs"]["y"] = y[y.columns[0]].ravel()
    cv_score_params["kwargs"]["estimator"] = estimator
    cv_score_params["kwargs"]["cv"] = cv_strategy
    scores = load_object(cv_score_params)
    mean_score = scores.mean()
    return mean_score
