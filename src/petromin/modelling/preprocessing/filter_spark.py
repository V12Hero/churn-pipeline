import logging
import typing as tp
import warnings

import pyspark
import pyspark.sql.functions as f

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def pre_processing(df: pyspark.sql.DataFrame, conditions: tp.List[str]) -> pyspark.sql.DataFrame:
    """
    Applies a series of filtering conditions to a PySpark DataFrame and returns
    the filtered DataFrame.

    This function allows for dynamic filtering of a DataFrame based on a list
    of conditions provided as strings. Each condition should be a valid SQL expression
    string understandable by PySpark. The DataFrame is then ordered by '_id' and
    '_observ_end_dt' columns.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame to be processed.
        conditions (List[str]): A list of string conditions to be applied as
        filters on the DataFrame.  Each string should be a valid SQL expression.

    Returns:
        pyspark.sql.DataFrame: The filtered DataFrame, sorted by '_id' and '_observ_end_dt'.

    Example:
        ```python
        from pyspark.sql import SparkSession

        # Initialize a Spark session
        spark = SparkSession.builder.appName("PreProcessing").getOrCreate()

        # Example DataFrame
        data = [(1, "2021-01-01"), (2, "2021-01-02"), (3, "2021-01-03")]
        df = spark.createDataFrame(data, ["_id", "_observ_end_dt"])

        # Example conditions
        conditions = ["_id > 1"]

        # Apply pre-processing
        processed_df = pre_processing(df, conditions)
        processed_df.show()
        ```
    """
    filter_cond = f.lit(True)
    if conditions:
        for cond in conditions:
            filter_cond = filter_cond & (f.expr(cond))

    df = df.filter(filter_cond).orderBy(["_id", "_observ_end_dt"]).fillna(0)
    logger.info(f"master table filtered shape: ({df.count()}, {len(df.columns)})")

    return df
