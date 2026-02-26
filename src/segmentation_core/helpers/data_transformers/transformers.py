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

"""This module contains helper functions.

This module contains helper functions for performing:
- categorical encoding
- imputation
- outlier removal

These helper functions allow the user to specify any sklearn-based transform in the
params.
"""
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Union

import pandas as pd
from kedro.utils import load_obj
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class SelectColumnTransformer(BaseEstimator, TransformerMixin):
    """Transformer that allows for the selection of a specified list of features.

    Attributes:
        features: List of features to be selected from a given set of data.
    """

    def __init__(self, features: List[str] = None):
        """Create a SelectColumnTransformer object."""
        self._features = list(set(features))

    def get_feature_names(self) -> List[str]:
        """Return the list of selected feature names.

        Returns:
            List of selected features
        """
        return self._features

    # pylint: disable=W0613
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> TransformerMixin:
        """Fit the transformer into the provided data.

        Args:
            X: input dataframe
            y: target variable series, only kept to conform to parent class
                signature

        Returns:
            TransformerMixin
        """
        if self._features is None:
            self._features = list(X.columns)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select columns from the provided dataframe.

        Select columns from the provided dataframe based on features identified in
        .fit().

        Args:
            X: input dataframe

        Returns:
            pd.DataFrame: contains only selected columns

        Raises:
            KeyError: When a feature is not found in provided dataframe
        """
        check_is_fitted(self, attributes=["_fitted"])
        missing_features = set(self._features) - set(X.columns)
        if missing_features:
            raise KeyError(f"The following features are missing: {missing_features}")

        return X[self._features]


class AbstractMLTransformer(BaseEstimator, TransformerMixin):
    """Base class which all model specific transformers should inherit or implement."""

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: pd.Series = None, **kwargs
    ) -> "AbstractMLTransformer":
        """Fit the transformer."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformer."""


def fit_transformer(
    train_data: pd.DataFrame, params: Dict[str, Any], features: List[str]
) -> Union[TransformerMixin, Pipeline]:
    """Fit a transformer into the training data provided.

    Args:
        train_data: data to be used to fit the transformer
        params: This is a set of parameters that must contain:
            - transformer_kwargs (Dict[str, Any]): params used to initialise the
                transformer object
            - transformer_fit_kwargs (Dict[str, Any]): params used for `.fit()` method
            - target_variable_col (str): This is the name of the target variable
                column in `train_data`
        features: This is a list of features to be fitted into the transformer

    Returns:
        fitted TransformerMixin

    Raises:
        ValueError: This is raised when the `class` is not specified in params
        KeyError: If `target_variable_col` is not in the data provided
    """
    if "class" not in params:
        raise ValueError("Must specify path to class of transformer")

    transformer_class = params["class"]
    transformer = load_obj(transformer_class)
    transformer_kwargs = params.get("transformer_kwargs", {})
    transformer_fit_kwargs = params.get("transformer_fit_kwargs", {})
    target_var_col = params.get("target_variable_col")

    target_var_series = None

    if target_var_col:
        if target_var_col not in train_data.columns:
            raise KeyError(f"Target variable {target_var_col} not in data")

        target_var_series = train_data[target_var_col]

    # If there is a specified list of features, we use a column selector
    # transformer to enable the transformer to remember the features involved
    # in the transform
    if features:
        transformer_pipeline = Pipeline(
            steps=[
                ("select_cols", SelectColumnTransformer(features=features)),
                ("actual_transform", transformer(**transformer_kwargs)),
            ]
        )
        return transformer_pipeline.fit(
            train_data, y=target_var_series, **transformer_fit_kwargs
        )

    return transformer(**transformer_kwargs).fit(
        train_data, y=target_var_series, **transformer_fit_kwargs
    )


# noinspection PyUnresolvedReferences
def apply_fitted_transformer(
    fitted_transformer: TransformerMixin,
    data: pd.DataFrame,
    features: List[str] = None,
    drop_other_features: bool = False,
) -> pd.DataFrame:
    """Apply a fitted sklearn-based transformer to a given dataset.

    Args:
        fitted_transformer: This is a trained sklearn-based transformer
        data: input data to apply the transformer to
        features: List of features included in the transform, this must be specified in
            the same order as in fit. If unspecified, the function assumes that the
            tranform will apply to all columns in the data provided
        drop_other_features: In the case that there are more features provided, this can
            be set to True so that the resulting dataframe only contains the necessary
            features. Otherwise, this function will return the transformed columns
            and the identity of columns not specified in features

    Returns:
        pd.DataFrame: contains the tranformed data
    """
    data_to_transform = data.copy()
    if features:
        data_to_transform = data[features]

    transformed_data = fitted_transformer.transform(data_to_transform)

    other_features = list(set(data.columns) - set(transformed_data.columns))
    if not drop_other_features and other_features:
        transformed_data = pd.concat([transformed_data, data[other_features]], 1)

    return transformed_data


def apply_transformer_multiple_sets(
    fitted_transformer: TransformerMixin,
    params: Dict[str, Any],
    *datasets: pd.DataFrame,
) -> List[pd.DataFrame]:
    """Apply a transformer to a given list of dataframes.

    Args:
        fitted_transformer: trained transformer object
        datasets: This is a list of dataframes to be transformed
        params: This is a dictionary containing the following:
            features (List[str]): list of features to apply the transformation on
            drop_other_features (bool = False: When set to True, this will drop columns
                in the dataframes that are not included in the transformation

    Returns:
        List of transformed pd.DataFrame
    """
    operator = apply_fitted_transformer
    transformed_datasets = list(
        map(lambda dataset: operator(fitted_transformer, dataset, **params), datasets)
    )

    return transformed_datasets


def apply_fitted_transformer_node(
    fitted_transformer: TransformerMixin, dataset: pd.DataFrame, params: Dict[str, Any]
):
    """Apply a fitted transformer to a single dataset.

    Args:
        fitted_transformer: trained transformer object
        dataset: This is the input dataframe
        params: This is a dictionary containing the following:
            features (List[str]): list of features to apply the transformation on
            drop_other_features (bool = False: When set to True, this will drop columns
                in the dataframes that are not included in the transformation

    Returns:
       pd.DataFrame: transformed input dataframe
    """
    return apply_fitted_transformer(fitted_transformer, dataset, **params)
