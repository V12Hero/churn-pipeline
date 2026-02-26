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

from typing import Dict, List

import pandas as pd

QUANTILES_COLOR_DICT = {
    "low": "#c42f37",
    "medium_low": "#e36939",
    "medium_high": "#e39139",
    "high": "#1b823e",
}


def _create_quantile_df(df: pd.DataFrame, columns: List[str]):
    """Create quantiles DFs.

    It takes a dataframe and a list of columns, and returns a dataframe with the quantiles of the
    columns

    Args:
      df (pd.DataFrame): The dataframe you want to create the quantile dataframe from.
      columns (List[str]): The columns to create quantiles for.

    Returns:
      A dataframe with the quantiles of the columns in the original dataframe.
    """
    quantile_df = pd.DataFrame(
        df[columns].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    ).T.rename(columns={0.0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"})

    cols = ["0", "0.25", "0.5", "0.75", "1.0"]

    for i in range(len(cols) - 1):
        quantile_df[f"{cols[i]}_{cols[i+1]}"] = quantile_df[
            [cols[i], cols[i + 1]]
        ].values.tolist()

    quantile_df.rename(
        columns={
            "0_0.25": "low",
            "0.25_0.5": "medium_low",
            "0.5_0.75": "medium_high",
            "0.75_1.0": "high",
        },
        inplace=True,
    )

    return quantile_df


def _add_quantile_col(value: float, quantile_dict: Dict):
    """Add quantile column.

    Adds quantile col based if the value is between in the range defined by the quantile dict.

    Args:
      value (Union[float, int]): The value to be added to the quantile column.
      quantile_dict (Dict): A dictionary of quantile ranges. The keys are the names of the quantile
    ranges, and the values are the ranges themselves.

    Returns:
      The quantile that the value falls into.
    """
    for k, v in quantile_dict.items():
        if type(v) == list:
            if v[0] <= value <= v[1]:
                return k


def _add_quantiles_to_df(
    input_df: pd.DataFrame, quantile_df: pd.DataFrame, columns: List[str]
):
    """Add quantiles to dataframe.

    It takes a dataframe and a dictionary of quantiles, and returns a new
    dataframe with a new column that says in which quantile (low, medium, high)
    the value is into.

    Args:
      input_df (pd.DataFrame): the dataframe that you want to add quantiles to
      quantile_df (pd.DataFrame): a dataframe with the quantile values for each column
      columns (List[str]): The columns to add quantiles to.

    Returns:
      A dataframe with the quantile column added.
    """
    df = input_df.copy()

    for col in columns:
        df[col, "quantile"] = input_df[col]["mean"].apply(
            lambda value: _add_quantile_col(value, quantile_df.loc[col].to_dict())
        )

    df.reset_index(inplace=True)

    return df


def _get_relative_difference_df(
    df: pd.DataFrame, cluster_col: str, columns: List[str], multi_index: bool = True
):
    """Compute the relative difference between features in a cluster.

    Returns a dataframe with the relative difference of the mean
    of each feature in the cluster compared to the population mean

    Args:
      df (pd.DataFrame): the dataframe you want to get the relative difference for
      cluster_col (str): the column name of the cluster labels
      columns (List[str]): the columns you want to get the relative difference for
      multi_index (bool): If True, the columns will be a multi-index with the column name and
    "relative_difference" as the second level. Defaults to True

    Returns:
      A dataframe with the relative difference of each cluster to the mean of the whole dataset.
    """
    # mean feature per cluster
    df_mean = pd.concat(
        [
            pd.DataFrame(df[columns].mean(), columns=["mean"]),
            df.groupby(cluster_col).mean()[columns].T,
        ],
        axis=1,
    )
    # normalize and get the absolute difference between the mean of the cluster and the whole dataset.
    df_relative_difference = (
        df_mean.apply(lambda x: round((x - x["mean"]) / x["mean"], 2) * 100, axis=1)
        .drop(columns=["mean"])
        .fillna(0.0)
    ).T

    if multi_index:
        df_relative_difference.columns = pd.MultiIndex.from_product(
            [df_relative_difference.columns, ["relative_difference"]]
        )

    return df_relative_difference


def _get_text_summary(
    cluster_df: pd.DataFrame, columns: List[str], pretty_md: bool = False
):
    """Creates a summary of the clusters features.

    Builds the text summary with the mean, quantile, std and variation from the mean
    for each feature in the cluster. Can return plain text or **pretty** markdown to display in notebooks.

    Args:
      cluster_df (pd.DataFrame): the dataframe of the cluster you want to get the summary for
      columns (List[str]): List of columns to be used for the summary.
      pretty_md (bool): If True, the text will be in markdown format. Defaults to False

    Returns:
      A string with the summary of the cluster
    """
    text = ""
    for col in columns:
        quantile_color = QUANTILES_COLOR_DICT.get(cluster_df[col]["quantile"].values[0])
        difference_color = (
            "red" if cluster_df[col]["relative_difference"].values[0] < 0 else "green"
        )
        deviation_sign = (
            "negative"
            if cluster_df[col]["relative_difference"].values[0] < 0
            else "positive"
        )
        if pretty_md:
            text = (
                text
                + "\n - This cluster belongs to "
                + f"<span style='color:{quantile_color}'> **{cluster_df[col]['quantile'].values[0]}** </span> quantile "
                + f"of the **{col}** variable "
                + f"with a **mean** value of {cluster_df[col]['mean'].values[0]:.2f}"
                + f", <span style='color:{difference_color}'> **{deviation_sign} deviation from the mean**</span> of "
                + f"{cluster_df[col]['relative_difference'].values[0]:.1f}%"
                + f", a **median** of {cluster_df[col]['median'].values[0]:.2f} "
                + f" and a **standard deviation** of {cluster_df[col]['std'].values[0]:.2f}"
            )
        else:
            text = (
                text
                + "\n - This cluster belongs to "
                + f"{cluster_df[col]['quantile'].values[0]} quantile "
                + f"of the {col} variable "
                + f"with a mean value of {cluster_df[col]['mean'].values[0]:.2f}"
                + f", deviation from the mean of {cluster_df[col]['relative_difference'].values[0]:.1f}%"
                + f", a median of {cluster_df[col]['median'].values[0]:.2f} "
                + f" and a standard deviation of {cluster_df[col]['std'].values[0]:.2f}"
            )
    text += "\n"

    return text
