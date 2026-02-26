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

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_area(geometry: gpd.GeoDataFrame, epsg: int) -> pd.Series:
    """Compute the area of each geometry in a GeoDataFrame..

    The first line of the docstring is a one sentence summary of the function

    Args:
      geometry (gpd.GeoDataFrame): the GeoDataFrame that you want to compute the area of.
      epsg (int): the epsg code of the CRS that the geometry is in.

    Returns:
      A pandas series with the area of each polygon in the geometry column.
    """
    geometry = geometry.to_crs(epsg)
    return geometry.area


def agg_cols(
    df: pd.DataFrame, agg_cols: List[str], agg_function: str = "sum"
) -> pd.Series:
    """Applies aggregation on a given set of columns.

    Args:
      df (pd.DataFrame): input_df
      agg_cols (List[str]): List with columns to aggregate
      agg_function (str): Name of aggregate function. Defaults to 'sum'. Defaults to sum

    Returns:
      A series with the aggregated values.
    """
    return df[agg_cols].agg(agg_function, axis=1)


def compute_mean_from_bins(df: pd.DataFrame, cols_values: Dict[str, float]):
    """Calculates the mean value of something based on bins from a dataframe
    and a dictionary saying what mean should be considered for each bin.

    Args:
      df (pd.DataFrame): the dataframe to be processed
      cols_values (Dict[str, float]): a dictionary of column names and their mean values

    Returns:
      The mean of the values in the bins.
    """
    new_df = df.copy()

    for col, mean_value in cols_values.items():
        new_df[f"{col}_tmp"] = new_df[col] * mean_value

    values = new_df[[f"{col}_tmp" for col in list(cols_values.keys())]].sum(axis=1)
    count = new_df[list(cols_values.keys())].sum(axis=1)

    return values / count


def add_percentage_cols(df: pd.DataFrame, cols: List[str]):
    """> It takes a dataframe and a list of columns, and returns a new
    dataframe with the same columns plus new columns that are the percentage of
    each column in the original list.

    Example:
        >> df = pd.DataFrame({'first_col': [2], 'other_col': [3]})
        >> add_percentage_cols(df, ['first_col', 'other_col'])

        first_col  other_col  first_col_per  other_col_per
        0          2          3            0.4            0.6

    Args:
      df (pd.DataFrame): the dataframe you want to add the percentage columns to
      cols (List[str]): The columns to add percentages for

    Returns:
      A new dataframe with the percentage of each column
    """
    new_df = df.copy()
    total = new_df[cols].sum(axis=1)

    for col in cols:
        new_df[f"{col}_per"] = new_df[col] / total

    return new_df


def add_multiple_percentage_cols(df: pd.DataFrame, cols_list: List[List[str]]):
    """> It takes a dataframe and a list of lists of column names, and returns
    a new dataframe with the percentage columns added.

    Args:
      df (pd.DataFrame): the dataframe you want to add the percentage columns to
      cols_list (List[List[str]]): a list of lists of column names. Each list of column names will be
    used to create a new column.

    Returns:
      A new dataframe with the percentage columns added.
    """
    new_df = df.copy()

    for cols in cols_list:
        new_df = add_percentage_cols(new_df, cols)

    return new_df


def list_features(data: pd.DataFrame, features: List[str]) -> pd.Series:
    """It takes a dataframe and a list of features, and converts the values of
    several columns on a list of the values.

    Args:
      data (pd.DataFrame): The dataframe that contains the features you want to list.
      features (List[str]): The list of features to be used in the model.

    Returns:
      A list of lists.
    """
    list_with_values = data[features].values.tolist()
    # Remove nan from list
    return [
        [value for value in values if not pd.isnull(value)]
        for values in list_with_values
    ]


def transform_to_timestamp(
    data: pd.DataFrame, column: str, format: Optional[str] = None
) -> pd.Series:
    """Transform a str timestamp to timestamp.

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


def time_delta(
    data: pd.DataFrame,
    start: Union[str, datetime],
    end: Union[str, datetime],
    units: str,
) -> pd.Series:
    """Returns the time difference between two columns on a dataframe.

    Args:
      data (pd.DataFrame): The dataframe containing the start and end columns
      start (Union[str, datetime]): The start date of the time period.
      end (Union[str, datetime]): The column name or datetime object that represents the end of the time
    period.
      units (str): The units of time to return.

    Returns:
      A series of time deltas
    """
    if isinstance(start, datetime) and isinstance(end, datetime):
        raise TypeError("start and/or end must be column names")

    if isinstance(start, str):
        start = data[start]

    if isinstance(end, str):
        end = data[end]

    return (
        pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)
    ) / np.timedelta64(1, units)
