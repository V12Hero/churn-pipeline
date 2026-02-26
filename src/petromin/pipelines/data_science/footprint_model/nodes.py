"""footprint model nodes"""
import logging
import typing as tp

import pandas as pd
import pyspark.sql.dataframe
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from petromin.modelling.evaluate.metrics import compute_regression_metrics
from petromin.modelling.hypertune.cross_validation import (
    hypertune_cross_validated_model,
)
from petromin.modelling.reproducibility.set_seed import seed_file
from petromin.python_utils.load.object_inyection import load_estimator

logger = logging.getLogger(__name__)


def footprint_model_hypertune(
    df: pyspark.sql.dataframe, params: tp.Dict[str, tp.Any]
) -> tp.Dict[str, tp.Any]:
    """
    Hyperparameter tuning for a machine learning model using cross-validation.

    Args:
        df (pyspark.sql.dataframe): The input pyspark DataFrame containing the dataset.
        params (Dict[str, Any]): A dictionary containing various parameters.
            - 'model_artifact' (Dict): A dictionary containing information about the model.
                - 'features' (List[str]): List of feature column names.
                - 'target' (str): The name of the target column.
            - Other hyperparameters and settings for hyperparameter tuning.

    Returns:
        Dict[str, Any]: A dictionary containing the results of hyperparameter tuning.
            - 'best_params' (Dict): The best hyperparameters found during tuning.
            - 'best_score' (float): The best score achieved during tuning.
            - Other relevant information about the tuning process.

    Notes:
        This function takes a DataFrame containing the dataset, extracts the specified
        features and target column, and performs hyperparameter tuning on a machine
        learning model using cross-validation.

        Example usage:
        ```python
        params = {
            "model_artifact": {
                "features": ["feature1", "feature2"],
                "target": "target_column"
            },
            "param1": value1,
            "param2": value2,
            # Add other hyperparameters and settings here
        }

        result = footprint_model_hypertune(data_df, params)
        print("Best hyperparameters:", result['best_params'])
        print("Best score:", result['best_score'])
        ```
    """
    seed_file()
    df = df.toPandas()

    # supervised learning definition
    features = params["model_artifact"]["features"]
    target = params["model_artifact"]["target"]

    X = df[features].astype(float)
    y = df[[target]].astype(float)

    hypertune_object = hypertune_cross_validated_model(
        X=X,
        y=y,
        params=params,
    )
    return hypertune_object


def footprint_model_fit(
    df: pyspark.sql.dataframe, best_params: tp.Dict[str, tp.Any]
) -> tp.Dict[str, tp.Any]:
    """
    Fit a machine learning model using the best hyperparameters.

    Args:
        df (psdf.DataFrame): The input PySpark DataFrame containing the dataset.
        best_params (Dict[str, Any]): A dictionary containing the best hyperparameters
            and model information.
            - 'model_artifact' (Dict): A dictionary containing information about the model.
                - 'features' (List[str]): List of feature column names.
                - 'target' (str): The name of the target column.
                - 'model' (str): The serialized model object to be loaded.

    Returns:
        Dict[str, Any]: A dictionary containing the trained machine learning model.
            - 'model' (Any): The trained machine learning model ready for predictions.

    Notes:
        This function takes a PySpark DataFrame containing the dataset, extracts the specified
        features and target column, loads the pre-trained machine learning model with the best
        hyperparameters, and fits the model to the data.

        Example usage:
        ```python
        best_hyperparameters = {
            "model_artifact": {
                "features": ["feature1", "feature2"],
                "target": "target_column",
                "model": "path/to/serialized_model.pkl"
            }
        }

        trained_model = footprint_model_fit(data_df, best_hyperparameters)
        predictions = trained_model.predict(new_data)
        ```
    """
    seed_file()
    df = df.toPandas()

    features = best_params["model_artifact"]["features"]
    target = best_params["model_artifact"]["target"]

    X = df[features].astype(float)
    y = df[[target]].astype(float)

    model = load_estimator(best_params["model_artifact"]["model"])
    model = model.fit(X, y)
    check_is_fitted(model)
    return model


def footprint_model_inference_train(
    df: pyspark.sql.dataframe, model: Pipeline, best_params: tp.Dict[str, tp.Any]
) -> pd.DataFrame:
    """
    Perform inference using a machine learning model on a DataFrame.

    Args:
        df (psdf.DataFrame): The input PySpark DataFrame containing the dataset.
        model (Pipeline): The trained machine learning model to be used for inference.
        best_params (Dict[str, Any]): A dictionary containing information about the best hyperparameters.
            - 'model_artifact' (Dict): A dictionary containing information about the model.
                - 'features' (List[str]): List of feature column names.

    Returns:
        pd.DataFrame: A DataFrame with inference results.
            - 'footprint_model_<model_name>' (Series): Predicted values generated by the model.

    Notes:
        This function takes a PySpark DataFrame containing the dataset, fills any missing
        values with 0.0, and uses a pre-trained machine learning model to perform inference
        on the data. The predictions are added as a new column in the DataFrame.

        Example usage:
        ```python
        best_hyperparameters = {
            "model_artifact": {
                "features": ["feature1", "feature2"]
            }
        }
        trained_model = load_model("path/to/trained_model")
        result_df = footprint_model_inference_train(data_df, trained_model, best_hyperparameters)
        ```
    """
    df = df.toPandas()

    # supervised learning definition
    features = best_params["model_artifact"]["features"]
    target = best_params["model_artifact"]["target"]
    model_name = model.__class__.__name__.lower()

    # features table
    X = df[features].astype(float)

    # inference
    df[f"footprint_model_{model_name}"] = model.predict(X)

    # compute regression metrics
    metrics = compute_regression_metrics(df[target], df[f"footprint_model_{model_name}"])

    return dict(
        df=df,
        metrics=metrics,
    )


def footprint_model_interpretability(model: Pipeline) -> tp.Dict[str, tp.Any]:
    """
    Interpret the feature importance of a machine learning model.

    Args:
        model (Pipeline): The trained machine learning model for which to calculate feature importance.

    Returns:
        Dict[str, Any]: A dictionary containing the feature importance information.
            - 'feature_importance' (pd.DataFrame): DataFrame with feature importance values.
                Columns: 'feature_importance' (importance scores)
                Index: Feature names.

    Notes:
        This function calculates and returns the feature importance of a trained machine
        learning model. The feature importance scores are sorted in descending order.

        Example usage:
        ```python
        trained_model = load_model("path/to/trained_model")
        interpretability_result = footprint_model_interpretability(trained_model)
        print("Feature Importance:")
        print(interpretability_result['feature_importance'])
        ```
    """
    try:
        feature_importance = pd.DataFrame(
            model.feature_importances_,
            index=model.feature_names_in_,
            columns=["feature_importance"],
        )
    except Exception:
        feature_importance = pd.DataFrame(
            model.coef_,
            index=model.feature_names_in_,
            columns=["feature_importance"],
        )
    feature_importance = feature_importance.sort_values(by=["feature_importance"], ascending=False)

    logger.info(feature_importance.head(30))
    # TODO: add more function to interpret the trees
    return dict(feature_importance=feature_importance)


def footprint_model_inference_test(df: pyspark.sql.dataframe, model: Pipeline) -> pd.DataFrame:
    """
    Perform inference using a machine learning model on a Geo DataFrame of Ecuador.

    Args:
        df (psdf.DataFrame): The input PySpark DataFrame containing the dataset.
        model (Pipeline): The trained machine learning model to be used for inference.

    Returns:
        pd.DataFrame: A DataFrame with inference results.
            - 'footprint_model_<model_name>' (Series): Predicted values generated by the model.

    Notes:
        This function takes a PySpark DataFrame containing the dataset, fills any missing
        values with 0.0, and uses a pre-trained machine learning model to perform inference
        on the data. The predictions are added as a new column in the DataFrame.

        Example usage:
        ```python
        trained_model = load_model("path/to/trained_model")
        result_df = footprint_model_inference_test(data_df, trained_model,)
        ```
    """
    # data reading + preprocessing
    df = df.toPandas()
    features = list(model.feature_names_in_)
    # TODO: validate if it's necessary to create this feature
    # create feature of min distance to the nearest point of interest
    columns = [col for col in df.columns if "min_distance" in col]
    df["distance_to_nearest_point"] = df[columns].min(axis=1)

    df[features] = df[features].astype(float)
    # model inference in the whole country
    df["footprint_model_prediction"] = model.predict(df[features])
    return df
