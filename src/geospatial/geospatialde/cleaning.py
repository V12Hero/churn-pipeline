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

import ast
import logging
import warnings
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from segmentation_core.helpers.data_processing.geometry import change_crs
from segmentation_core.helpers.data_transformers.cleaning_utils import (
    _replace_elements,
    _unidecode_strings,
)
from segmentation_core.helpers.tag_managment.tag_dict import TagDict

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def change_names(
    data: pd.DataFrame, td: TagDict, source: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Change the name of the dataframe columns."""
    td.filter(condition={"source": source, "derived": False})
    data.columns = [td[col]["name"] for col in data.columns]

    return data


def change_dtype(
    data: pd.DataFrame, td: TagDict, source: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Change the columns types according to the td."""
    td.filter(condition={"source": source, "derived": False})
    for col in data.columns:
        logger.info(f"Changing type of {col}")
        dtype = td[col]["data_type_new"]
        data[col] = data[col].astype(dtype=dtype, errors="raise")

    return data


def change_epsg(data: pd.DataFrame, epsg: int) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Change the crs of the given data for the epsg."""
    if not isinstance(data, gpd.GeoDataFrame):
        return data
    return change_crs(data, epsg)


def clip_data(
    data: pd.DataFrame, td: TagDict, source: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Clip dfs according to a range.

    It clips the data in the dataframe `data` to the min and max values specified in the tag dictionary
    `td` for the source `source`

    Args:
      data (pd.DataFrame): The dataframe to be clipped
      td (TagDict): TagDict
      source (str): The source of the data.

    Returns:
      A dataframe with the clipped data.
    """
    td.filter(condition={"source": source, "derived": False})

    for col in data.columns:
        if td[col]["clip_min"] is not None or td[col]["clip_max"] is not None:
            data[col] = np.clip(
                data[col], a_min=td[col]["clip_min"], a_max=td[col]["clip_max"]
            )

    return data


def filter_data(
    data: pd.DataFrame, td: TagDict, source: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Filter the data bvased on allowed_values"""

    td.filter(condition={"source": source, "derived": False})

    for col in data.columns:
        # Catch error in literal_eval. If the error is becasuse of an nan do not raise
        if td[col]["allowed_values"]:
            allowed_values = ast.literal_eval(td[col]["allowed_values"])
            data = data.loc[data[col].isin(allowed_values), :]

    return data


def _trim_column(
    series: pd.Series,
    min_value: Union[float, int] = None,
    max_value: Union[float, int] = None,
) -> pd.Series:
    """Convert out of range values into NaN in a column."""

    if min_value is not None:
        series.loc[series < min_value] = np.nan
    if max_value is not None:
        series.loc[series > max_value] = np.nan
    return series


def range_data(
    data: pd.DataFrame, td: TagDict, source: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Convert out of range values into NaN in a dataframe."""

    td.filter(condition={"source": source, "derived": False})

    for col in data.columns:
        data[col] = _trim_column(
            data[col].copy(), td[col]["range_min"], td[col]["range_max"]
        )

    return data


def remove_missing_values(
    data: pd.DataFrame, td: TagDict, source: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Remove missing values.

    This function takes a dataframe and a tag dictionary and replaces the missing values in the
    dataframe with NaN

    Args:
      data (pd.DataFrame): The dataframe you want to clean
      td (TagDict): TagDict
      source (str): The source of the data. This is used to filter the TagDict.

    Returns:
      A dataframe with the missing values removed.
    """
    td.filter(condition={"source": source, "derived": False})
    for col in data.columns:
        # Catch error in literal_eval. If the error is becasuse of an nan do not raise
        if td[col]["missing_values"]:
            nas = ast.literal_eval(td[col]["missing_values"])
            data.loc[:, col] = data[col].replace(to_replace=nas, value=np.nan)
    return data


def standarize_string_formats_dataframes(
    df: pd.DataFrame, columns: List
) -> pd.DataFrame:
    """standarize string formats.

    For each string a in df column. Standarize string formats.

    example:
        input: tienda del barrío de ñandú
        output: tienda_del_barrio_de_nandu

    Args:
      df (pd.DataFrame): the dataframe you want to unidecode
      columns (List): List of columns to standarize

    Returns:
      A dataframe with the columns specified in the columns parameter unidecoded.
    """
    if len(columns) > 0:
        for col in columns:
            df[col] = df[col].apply(lambda col: _unidecode_strings(col))
            df[col] = df[col].apply(lambda col: _replace_elements(col))
    return df


def map_values(data: pd.DataFrame, td: TagDict, source: str) -> pd.DataFrame:
    """Map the values of the columns according to the td."""
    td.filter(filter_col="source", condition=source)

    # Get the mapping keys
    map_keys = [k for k in td.keys() if k.startswith("mapping_")]

    # Iterate columns
    for col in data.columns:
        # Iterate mappings
        for key in map_keys:
            # If mapping exists
            if td[col][key] is not None:
                # Get the new column name from td
                data[td[col][f"name_{key}"]] = data[col].map(
                    ast.literal_eval(td[col][key])
                )

    return data


def _pivot_series(series: pd.Series) -> pd.DataFrame:
    """Pivot a series to True or False values."""
    name = series.name
    columns = series.dropna().unique()
    df = series.to_frame(name=name)
    for col in columns:
        df[f"{name}_{col}"] = series == col

    return df.drop(name, axis=1)


def pivot_values(data: pd.DataFrame, pivot: List[str]) -> pd.DataFrame:
    """Pivot the selected dataframe columns to a wide format using True and False."""
    if pivot is None:
        return data

    for col in pivot:
        data = pd.concat([data, _pivot_series(data[col])], axis=1)

    return data


def remove_columns(data: pd.DataFrame, td: TagDict, source: str) -> pd.DataFrame:
    """Remove columns from the dataframe according to the td."""
    td.filter(condition={"source": source})

    logger.info(f'Removing columns: {td.select(filter_col="drop", condition=True)}')

    return data.drop(td.select(filter_col="drop", condition=True), axis=1)


def transform_to_timestamp(
    data: pd.DataFrame, column: str, format: Optional[str] = None
) -> pd.Series:
    """
    Transform a str timestamp to timestamp

    Args:
      data (pd.DataFrame): input dataframe
      column (str): name of column to convert
      format (Optional[str]): The format of the input string.

    Returns:
      A series of timestamps
    """
    logger.info(f"Timestamp {column}")
    logger.info(f"Data columns: {list(data.columns)}")
    return pd.to_datetime(data[column], format=format, errors="coerce")


def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    df = df.query(f"country == '{country}'")
    return df


def filter_nulls(df: pd.DataFrame, td: TagDict, source: str):
    """Filter null rows based on tag dict configuration."""
    td.filter(condition={"source": source})
    for col in df.columns:
        if td[col]["filter_nulls"]:
            logger.info(f"Filtering rows with nulls in {col}")
            df = df.query(f"{col}.notna()")

    return df


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop pandas duplicates."""
    data = data.drop_duplicates()
    return data
