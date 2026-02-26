"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import typing as tp

import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window
from feature_generation.v1.nodes.features.create_column import (
    create_columns_from_config,
)


def create_ftr_windows_transactions(
    input_data: pyspark.sql.DataFrame, instructions: tp.Dict, sequential: tp.Dict, params_keep_cols: None
) -> pyspark.sql.DataFrame:
    """Return transaction features."""

    features = create_columns_from_config(input_data, instructions, sequential, params_keep_cols)

    return features.orderBy(["_id", "_observ_end_dt"])


def create_ftr_geolocation(
    input_data: pyspark.sql.DataFrame, instructions: tp.Dict, sequential: tp.Dict
) -> pyspark.sql.DataFrame:
    """Return transaction features."""

    features = create_columns_from_config(input_data, instructions, sequential)

    return features.orderBy(["_id", "_observ_end_dt"])
