import typing as tp

import pyspark
import pyspark.sql.functions as f


def create_footprint_normalized_target(
    df: pyspark.sql.DataFrame, params: tp.List[str]
) -> pyspark.sql.DataFrame:
    """
    Create a normalized target column in a PySpark DataFrame by dividing two columns.

    Args:
        df (pyspark.sql.DataFrame): The input PySpark DataFrame.
        params (dict): A dictionary containing the following keys:
            - "numerator" (str): The name of the numerator column.
            - "denominator" (str): The name of the denominator column.
            - "target_name" (str): The name of the target (result) column.

    Returns:
        pyspark.sql.DataFrame: A DataFrame with the normalized target column added.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("example").getOrCreate()
        >>> data = [(1, 10, 2), (2, 20, 4), (3, 30, 6)]
        >>> columns = ["id", "numerator", "denominator"]
        >>> df = spark.createDataFrame(data, columns)
        >>> params = {
        ...     "numerator": "numerator",
        ...     "denominator": "denominator",
        ...     "target_name": "result_column"
        ... }
        >>> result_df = create_footprint_normalized_target(df, params)
        >>> result_df.show()
        +---+---------+-----------+--------------+
        | id|numerator|denominator|result_column|
        +---+---------+-----------+--------------+
        |  1|       10|          2|           5.0|
        |  2|       20|          4|           5.0|
        |  3|       30|          6|           5.0|
        +---+---------+-----------+--------------+
    """
    numerator = params["numerator"]
    denominator = params["denominator"]
    target_name = params["target"]
    return df.withColumn(target_name, f.col(numerator) / f.col(denominator))
