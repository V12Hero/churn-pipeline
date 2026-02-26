import typing as tp

import pandas as pd


def get_distribution_and_relative_difference_report(
    df: pd.DataFrame, cluster_col: str, select_cols_summary: tp.List[str]
) -> pd.DataFrame:
    """
    Generates a report containing the distribution and relative differences of selected
    columns in a DataFrame, grouped by a specified cluster column.

    This function calculates the mean and median for the selected columns in the DataFrame,
    both overall and for each cluster. It then computes the relative differences of these
    values from the overall mean and median. The report is returned as a dictionary of
    DataFrames, with keys for mean, median, and their relative differences.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be analyzed.
        cluster_col (str): The name of the column in df used for clustering.
        select_cols_summary (List[str]): A list of column names in df to include in the summary.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the following DataFrames:
            - 'mean': Mean values of select_cols_summary for each cluster and overall.
            - 'relative_difference_mean': Relative differences from the overall mean for each cluster.
            - 'median': Median values of select_cols_summary for each cluster and overall.
            - 'relative_difference_median': Relative differences from the overall median for each cluster.

    Example:
        ```python
        import pandas as pd

        # Example DataFrame
        data = {'cluster': [1, 1, 2, 2, 3, 3],
                'feature1': [10, 15, 10, 20, 30, 25],
                'feature2': [100, 110, 90, 85, 120, 115]}
        df = pd.DataFrame(data)

        # Getting the report
        report = get_distribution_and_relative_difference_report(
            df, cluster_col='cluster', select_cols_summary=['feature1', 'feature2']
        )
        print(report['mean'])
        print(report['relative_difference_mean'])
        ```
    """
    # cluster size assignment
    df[cluster_col] = df[cluster_col].astype(str)
    df = df[[cluster_col] + select_cols_summary]

    cluster_size = pd.DataFrame(df.groupby(cluster_col).size(), columns=["cluster_size"]).T

    # mean and median preprocessing
    df_mean = pd.concat(
        [
            pd.DataFrame(df.mean(), columns=["mean"]),
            df.groupby(cluster_col).mean().T,
        ],
        axis=1,
    )
    df_median = pd.concat(
        [
            pd.DataFrame(df.median(), columns=["median"]),
            df.groupby(cluster_col).median().T,
        ],
        axis=1,
    )

    # relatives differences
    df_relative_difference_mean = (
        df_mean.apply(lambda x: round((x - x["mean"]), 4), axis=1)
        .drop(columns=["mean"])
        .fillna(0.0)
    )
    df_relative_difference_median = (
        df_median.apply(lambda x: round((x - x["median"]), 4), axis=1)
        .drop(columns=["median"])
        .fillna(0.0)
    )

    # index column name
    cluster_size.index.name = "feature"
    df_mean.index.name = "feature"
    df_relative_difference_mean.index.name = "feature"
    df_median.index.name = "feature"
    df_relative_difference_median.index.name = "feature"

    # concat
    df_mean = pd.concat([cluster_size, df_mean])
    df_median = pd.concat([cluster_size, df_median])

    output_dict = dict(
        mean=df_mean.reset_index(),
        relative_difference_mean=df_relative_difference_mean.reset_index(),
        median=df_median.reset_index(),
        relative_difference_median=df_relative_difference_median.reset_index(),
    )
    return output_dict
