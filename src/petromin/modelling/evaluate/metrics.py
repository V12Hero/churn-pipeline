import typing as tp

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def transform_output_dict(
    output_dict: tp.Dict[str, tp.Any]
) -> tp.Dict[str, tp.Union[float, tp.List[float]]]:
    transformed_dict = {}

    for key, value in output_dict.items():
        if isinstance(value, list):
            transformed_value = [{"value": v, "step": i + 1} for i, v in enumerate(value)]
        else:
            transformed_value = {"value": value, "step": 1}

        transformed_dict[key] = transformed_value

    return transformed_dict


def compute_regression_metrics(
    target: np.array,
    prediction: np.array,
) -> dict[str, np.array]:
    """Compute regression metrics.

    Args:
        target (np.array): Numpy real price.
        prediction (np.array): Predicted Price.
        horizon (str): The name of the metric according to
        the horizon.

    Returns:
        tp.Dict[str, np.array]: Price prediction metrics.
    """
    output_dict = {
        "r2": r2_score(target, prediction),
        "mae": mean_absolute_error(target, prediction),
        "rmse": root_mean_square_error(target, prediction),
        "mse": mean_squared_error(target, prediction),
        "rdse": root_standard_deviation_square_error(target, prediction),
        "mape": mean_absolute_percentage_error(target, prediction) * 100,
        "mean_target": calculate_statistic(target, statistic="mean"),
        "mean_prediction": calculate_statistic(prediction, statistic="mean"),
        "median_target": calculate_statistic(target, statistic="median"),
        "median_prediction": calculate_statistic(prediction, statistic="median"),
    }
    output_dict = transform_output_dict(output_dict)
    return output_dict


def calculate_statistic(data: tp.Union[pd.DataFrame, np.array], statistic="mean"):
    """
    The function `calculate_statistic` calculates either the mean or median of a given dataset.

    Args:
      data (tp.Union[pd.DataFrame, np.array]): The `data` parameter can be either a pandas DataFrame or
    a numpy array. It represents the dataset for which you want to calculate the statistic.
      statistic: The `statistic` parameter is a string that specifies the type of statistic to
    calculate. It can be either 'mean' or 'median'. Defaults to mean

    Returns:
      the calculated statistic value, either the mean or median, depending on the input.
    """
    if statistic == "mean":
        return np.mean(np.array(data))
    elif statistic == "median":
        return np.median(np.array(data))
    else:
        raise ValueError("Invalid statistic. Choose 'mean' or 'median'.")


def root_standard_deviation_square_error(y_true: np.array, y_pred: np.array) -> float:
    """Root standard deviation of error.

    This function calculates the root standard deviation of the error
    between the true and predicted values of a given dataset.


    Args:
      y_true (np.array): The true values of the target variable
      (or dependent variable) in a regression problem.
      These are the actual values that we are trying to predict.
      y_pred (np.array): The predicted values of the target variable.


    Returns:
      the square root of the standard deviation of the error between the
      true values (y_true) and predicted values (y_pred) of a
      regression model.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.reshape(y_true, (1, -1))
    y_pred = np.reshape(y_pred, (1, -1))
    return np.sqrt(np.std((y_true - y_pred) ** 2))


def root_mean_square_error(y_true: np.array, y_pred: np.array) -> float:
    """Root mean square error.

    This function calculates the root mean square error
    between the true and predicted values of a given dataset.


    Args:
      y_true (np.array): The true values of the target variable
      (or dependent variable) in a regression problem.
      These are the actual values that we are trying to predict.
      y_pred (np.array): The predicted values of the target variable.


    Returns:
      the square root mean square error between the
      true values (y_true) and predicted values (y_pred) of a
      regression model.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.reshape(y_true, (1, -1))
    y_pred = np.reshape(y_pred, (1, -1))
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
