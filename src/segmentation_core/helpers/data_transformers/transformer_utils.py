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

"""Contains nodes that support the use of transformers in a Kedro pipeline."""
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from segmentation_core.helpers.data_dict.generator import DataDict
from segmentation_core.helpers.data_transformers import cleaning_utils

logger = logging.getLogger(__name__)


def enforce_schema_using_dict(data: pd.DataFrame, data_dict: DataDict) -> pd.DataFrame:
    """Enforce a schema on a dataframe using a data dictionary.

    Args:
        data: input dataframe
        data_dict: DataDict object

    Returns:
        pd.DataFrame with data types defined per col
    """
    df_typed = data.copy()
    categorical_features = data_dict.get_features(data_type="categorical")
    numeric_features = data_dict.get_features(data_type="numeric")
    datetime_features = data_dict.get_features(data_type="datetime")
    boolean_features = data_dict.get_features(data_type="boolean")

    categoricals = {feat: "categorical" for feat in categorical_features}
    numerics = {feat: "numeric" for feat in numeric_features}
    datetimes = {feat: "datetime" for feat in datetime_features}
    booleans = {feat: "boolean" for feat in boolean_features}

    feature_grps = [categoricals, numerics, datetimes, booleans]

    data_types = {}
    for dict_ in feature_grps:
        for col, dtype in dict_.items():
            data_types[col] = dtype

            if col not in data.columns:
                logger.warning(f"{col} is not in the provided dataframe")

    df_typed = cleaning_utils.enforce_custom_schema(df_typed, data_types)
    return df_typed


def deduplicate_df_using_dict(
    data: pd.DataFrame, data_dict: DataDict, params: Dict[str, Any] = None
) -> pd.DataFrame:
    """Deduplicate a pandas dataframe.

    Deduplicate a pandas dataframe using the primary key defined in the data dictionary.

    Args:
        data: input dataframe
        data_dict: DataDict object
        params: Dict containing keyword params for `pd.drop_duplicates`

    Returns:
        pd.DataFrame without duplicates based on defined primary key
    """
    primary_key = data_dict.get_key_column()
    params = params if params else {}
    df_deduped = cleaning_utils.deduplicate_pandas(data, subset=primary_key, **params)
    return df_deduped


def get_features_to_impute(params: Dict[str, Any], data_dict: DataDict) -> List[str]:
    """Determine which features are to be imputed.

    Args:
        params: This is a set of parameters for imputation. The following
            keywords are expected:
             "class": path to the class that contains the implementation
                of the transformer
            "target_variable_col": The name of the target variable column
            "transformer_kwargs": These are key-value pairs defining the keyword
                params to initialise the transformer
            "features": This is a list of features to impute
            "data_type": This can be set to either `categorical` or `numeric`
        data_dict: DataDict object. If the features are not specified

    Returns:
        List (str): This is a list of features to impute

    Raises:
        ValueError: This is raised when the data type provided is not
        `numeric` or `categorical`
    """
    if params["data_type"] not in ["numeric", "categorical"]:
        raise ValueError("Only `numeric` or `categorical` data can be imputed.")

    features_to_impute = params.get("features", None)

    if features_to_impute is None:
        features_to_impute = data_dict.get_features(data_type=params["data_type"])

    return features_to_impute


def get_cols_to_skip(data_dict: DataDict) -> List[Optional[str]]:
    """Extract the key column of the master table.

    This function extracts the key column of the master table that will be skipped when
    imputing the data.

    Args:
        data_dict: DataDict object

    Returns:
        Optional(str): This is a field to skip
    """
    return [data_dict.get_key_column()] or []


def get_model_input_datasets(inputs: Dict[str, Any]) -> List[str]:
    """Validate and returns the name of the input datasets present.

    Validates and returns the name of the input datasets present within the inputs
    configuration of the modular pipeline.

    Args:
        inputs (Dict[str, Any]): node input mapping dictionary

    Raises:
        ValueError: if train_data, test_data are not present in the inputs

    Returns:
        List of sets (train, test, valid) mapped from the inputs dictionary
    """
    mandatory_inputs = {"train_data", "test_data"}

    if inputs is None:
        return list(mandatory_inputs)

    mandatory_inputs_diff = mandatory_inputs - set(inputs.keys())

    if len(mandatory_inputs_diff) > 0:
        raise ValueError(
            "Imputation pipeline inputs must contain `train_data`, "
            "`test_data` and optionally `valid_data`"
        )

    full_inputs_set = mandatory_inputs.copy()
    full_inputs_set.add("valid_data")

    datasets = [ipt_set for ipt_set in full_inputs_set if inputs.get(ipt_set)]

    return datasets
