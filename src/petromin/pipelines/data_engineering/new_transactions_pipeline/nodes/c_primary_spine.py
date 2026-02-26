"""Primary layer nodes."""

import logging
import typing as tp
from datetime import date
from math import radians
from string import ascii_lowercase
from typing import List, Union

import pandas as pd
import pyspark.sql.dataframe
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)


def create_spine(
    customers_info: pyspark.sql.DataFrame,
    vehicles_info: pyspark.sql.DataFrame,
    transactions_info: pyspark.sql.DataFrame,
    start_dt: date,
    end_dt: date,
    frequency: str,
    customers_to_remove: list,
) -> pyspark.sql.DataFrame:
    """Return a spine DataFrame with patterned '_id' and '_observ_end_dt' columns.

    Args:
        customers_info (pyspark.sql.DataFrame): DataFrame containing customer information.
        start_dt (date): Start date for the spine generation.
        end_dt (date): End date for the spine generation.
        frequency (str): Frequency for date generation (e.g., 'D' for daily, 'M' for monthly).
        customers_to_remove (list): List of customer IDs to be removed from the spine.

    Returns:
        pyspark.sql.DataFrame: A DataFrame representing the spine with '_id' and '_observ_end_dt' columns.

    Examples:
        Usage examples of the create_spine function:

        1. Create a daily spine for all customers, removing specific customers:
        ```python
        from pyspark.sql import SparkSession
        from datetime import date

        spark = SparkSession.builder.getOrCreate()

        customers_info = spark.createDataFrame([(1,), (2,), (3,), (4,)], ['customer_id'])
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 5)
        frequency = 'D'
        customers_to_remove = [2, 4]

        spine = create_spine(customers_info, start_date, end_date, frequency, customers_to_remove)
        spine.show()
        ```

        2. Create a monthly spine for all customers, keeping all customers:
        ```python
        from pyspark.sql import SparkSession
        from datetime import date

        spark = SparkSession.builder.getOrCreate()

        customers_info = spark.createDataFrame([(1,), (2,), (3,), (4,)], ['customer_id'])
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        frequency = 'M'
        customers_to_remove = []

        spine = create_spine(customers_info, start_date, end_date, frequency, customers_to_remove)
        spine.show()
        ```
    """
    spark = SparkSession.builder.getOrCreate()

    # Create range dataframe
    range_of_dates = pd.DataFrame()
    range_of_dates["_observ_end_dt"] = pd.date_range(start=start_dt, end=end_dt, freq=frequency)

    logger.info(f"Spine range: {range_of_dates["_observ_end_dt"].min()} - {range_of_dates["_observ_end_dt"].max()}")

    # Create a DataFrame with the range of dates
    range_of_dates = spark.createDataFrame(range_of_dates).withColumn(
        "_observ_end_dt", f.date_format("_observ_end_dt", "yyyy-MM-dd")
    ).withColumn(
        "_observ_end_dt",
        f.to_date(f.last_day(f.col("_observ_end_dt")))
        # f.to_date(f.date_trunc("quarter", f.col("_observ_end_dt")))
    )

    # breakpoint()

    # Get the distinct customers from customers_info
    customers_info = customers_info.select("customer_id", "mobile").distinct()
    vehicles_info = vehicles_info.select("customer_id", "customer_vehicle_id", "plate_number").distinct()

    customer_universe_agg  = vehicles_info.groupBy(
        "customer_id"
    ).agg(
        f.countDistinct("customer_vehicle_id").alias("count")
    )

    customer_universe = customers_info.join(
        vehicles_info,
        on="customer_id",
        how="outer",
    ).join(
        customer_universe_agg,
        on="customer_id",
        how="left",
    ).filter(
        f.col("count") < 6
    ).drop("count")


    # Filter out customers to remove
    if customers_to_remove is not None:
        customer_universe = customer_universe.filter(~f.col("customer_id").isin(customers_to_remove))

    wrong_plates = [
        f"{number}"*4 + f"{letter.upper()}"*3 
        for letter in ascii_lowercase 
        for number in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]

    wrong_mobiles = [
        f"0{number1}" + f"{number2}"*8 
        for number1 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
        for number2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]

    customer_universe = customer_universe.filter(
        f.length("plate_number") == 7
    ).filter(
        f.length("mobile") == 12
    ).filter(
        ~f.col("plate_number").isin(wrong_plates)
    ).filter(
        ~f.col("mobile").isin(wrong_mobiles)
    ).withColumn(
    #     "mobile",
    #     f.overlay(f.col("mobile"), f.lit("966"), 0, 2)
    # ).withColumn(
        "_id",
        f.concat_ws("__", "plate_number", "mobile")
    ).withColumn(
        "_key", 
        f.concat_ws("__", "customer_id", "customer_vehicle_id")
    ).select("_id", "_key")

    # breakpoint()

    transactions_info = create_auxillary_columns(
        transactions_info,
        unit_of_analysis=["customer_id", "customer_vehicle_id"],
        time_column="transaction_dt"
    ).withColumnRenamed("_id", "_key")

    transactions_info_agg = transactions_info.groupBy(
        "_key"
    ).agg(
        f.min("_observ_end_dt").alias("first_transaction_period")
    )

    transactions_info_agg = transactions_info_agg.join(
        customer_universe,
        on="_key",
        how="inner"
    ).groupBy(
        "_id"
    ).agg(
        f.min("first_transaction_period").alias("first_transaction_period")
    ).select("_id", "first_transaction_period")

    # Create the spine with a cross join
    spine = customer_universe.crossJoin(
        range_of_dates
    ).join(
        transactions_info_agg,
        on="_id",
        how="left"
    ).filter(
        f.col("_observ_end_dt") >= f.col("first_transaction_period")
    )

    # breakpoint()

    # Order dataframe by ids and cohort date
    spine = spine.select("_id", "_key", "_observ_end_dt").orderBy(["_id", "_key", "_observ_end_dt"])

    return spine


def create_auxillary_columns(
    df: pyspark.sql.DataFrame, unit_of_analysis: Union[str, list[str]] = None, time_column: str = None
) -> pyspark.sql.DataFrame:
    """Return a DataFrame with patterned '_id' and '_observ_end_dt' columns.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame.
        unit_of_analysis (str, optional): The column used for '_id' pattern. Default is None.
        time_column (str, optional): The column used for '_observ_end_dt' pattern. Default is None.

    Returns:
        pyspark.sql.DataFrame: A DataFrame with '_id' and '_observ_end_dt' columns as specified.

    Examples:
        Usage examples of the create_auxiliary_columns function:

        1. Add '_id' and '_observ_end_dt' columns based on a unit of analysis and a time column:
        ```python
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        data = [(1, '2023-01-01'), (2, '2023-02-01'), (3, '2023-03-01')]
        columns = ['unit_id', 'date']

        df = spark.createDataFrame(data, columns)

        result_df = create_auxiliary_columns(df, unit_of_analysis='unit_id', time_column='date')
        result_df.show()
        ```

        2. Add only '_id' column based on a unit of analysis:
        ```python
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        data = [(1, '2023-01-01'), (2, '2023-02-01'), (3, '2023-03-01')]
        columns = ['unit_id', 'date']

        df = spark.createDataFrame(data, columns)

        result_df = create_auxiliary_columns(df, unit_of_analysis='unit_id')
        result_df.show()
        ```

        3. Add only '_observ_end_dt' column based on a time column:
        ```python
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        data = [(1, '2023-01-01'), (2, '2023-02-01'), (3, '2023-03-01')]
        columns = ['unit_id', 'date']

        df = spark.createDataFrame(data, columns)

        result_df = create_auxiliary_columns(df, time_column='date')
        result_df.show()
        ```

        4. No additional columns added (both unit_of_analysis and time_column are None):
        ```python
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        data = [(1, '2023-01-01'), (2, '2023-02-01'), (3, '2023-03-01')]
        columns = ['unit_id', 'date']

        df = spark.createDataFrame(data, columns)

        result_df = create_auxiliary_columns(df)
        result_df.show()
        ```
    """
    if unit_of_analysis:
        if len(unit_of_analysis) < 2:
            df = df.withColumn(
                "_id", 
                f.col(unit_of_analysis)
            )
        else:
            df = df.withColumn(
                "_id", 
                f.concat_ws("__", *unit_of_analysis)
            )

    if time_column:
        df = df.withColumn(
            "_observ_end_dt",
            f.last_day(f.col(time_column))
            # f.to_date(f.date_trunc("quarter", f.col(time_column)))
        )

    return df
