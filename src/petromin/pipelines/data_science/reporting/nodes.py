"""Reporting nodes."""
import typing as tp

import pandas as pd

from petromin.reporting.excel import get_distribution_and_relative_difference_report


def compute_segmentation_excel_report(
    df: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
    model_params: tp.Dict[str, tp.Any],
) -> tp.Dict[str, tp.Any]:
    """
    Generates a segmentation report for an Excel file, based on distribution and relative difference metrics.

    This function takes a DataFrame and parameters dict as input. It uses these parameters to
    determine which columns to summarize and the column used for clustering. The function then
    generates a report using `get_distribution_and_relative_difference_report`. The report includes
    dataframes for mean, median, and their relative differences for the selected columns, grouped
    by the specified clustering column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be analyzed.
        params (Dict[str, Any]): A dictionary containing parameters for the report generation.
                                 This includes keys for 'excel_report' which further includes
                                 'summary_columns', and 'cluster_col' indicating the clustering column.

    Returns:
        Dict[str, Any]: A dictionary containing the segmentation report. The report is structured as
                        returned by `get_distribution_and_relative_difference_report`.

    Example:
        ```python
        import pandas as pd

        # Example DataFrame
        data = {'cluster': [1, 1, 2, 2, 3, 3],
                'feature1': [10, 15, 10, 20, 30, 25],
                'feature2': [100, 110, 90, 85, 120, 115]}
        df = pd.DataFrame(data)

        # Parameters for report generation
        params = {
            'excel_report': {'summary_columns': ['feature1', 'feature2']},
            'cluster_col': 'cluster'
        }

        # Generating the report
        report = compute_segmentation_excel_report(df, params)
        print(report['mean'])
        print(report['relative_difference_mean'])
        ```
    """
    select_cols_summary = params["excel_report"]["summary_columns"]
    cluster_col = params["cluster_col"]

    features = sorted(
        set([name for sublist in model_params["features"].values() for name in sublist])
    )

    report = get_distribution_and_relative_difference_report(
        df=df, select_cols_summary=select_cols_summary, cluster_col=cluster_col
    )
    report["clusters"] = df[["store_id", "cluster_id"] + features]
    return report
