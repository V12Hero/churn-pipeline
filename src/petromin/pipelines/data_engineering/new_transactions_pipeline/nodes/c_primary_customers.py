"""Primary layer nodes."""

import logging
import typing as tp
from datetime import date
from math import radians
from typing import List

import numpy as np
import pandas as pd
import pyspark.sql.dataframe
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from sklearn.neighbors import BallTree

from .c_primary_spine import create_auxillary_columns

logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()


def create_prm_customers(
    customers_info: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """Return the primary customers table.

    Args:
        spine (pyspark.sql.DataFrame): The spine DataFrame.
        customers_info (pyspark.sql.DataFrame): DataFrame containing customers information.

    Returns:
        pyspark.sql.DataFrame: The primary customers table.

    Examples:
        Usage examples of the create_prm_customers function:

        1. Create a primary customers table from spine and customers_info DataFrames:
        ```python
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col

        spark = SparkSession.builder.getOrCreate()

        # Sample data for customers_info DataFrame
        customers_info_data = [
            (1, "customer A", "Location A", "Region A", "Classification A", "2023-01-01", 40.123, -75.456),
            (2, "customer B", "Location B", "Region B", "Classification B", "2023-02-01", 41.234, -76.567),
            (3, "customer C", "Location C", "Region C", "Classification C", "2023-03-01", 42.345, -77.678),
        ]
        customers_info_columns = [
            "customer_id", "customer_name", "customer_location", "region_name",
            "customer_classification", "customer_start_date", "latitude", "longitude"
        ]
        customers_info_df = spark.createDataFrame(customers_info_data, customers_info_columns)

        # Sample data for spine DataFrame
        spine_data = [
            (1, "2023-01-01"), (2, "2023-02-01"), (3, "2023-03-01"),
            (1, "2023-02-01"), (2, "2023-03-01"), (3, "2023-04-01"),
        ]
        spine_columns = ["_id", "_observ_end_dt"]
        spine_df = spark.createDataFrame(spine_data, spine_columns)

        # Create the primary customers table
        prm_customers = create_prm_customers(spine_df, customers_info_df)
        prm_customers.show()
        ```

        2. Create a primary customers table with missing customer size values filled using the median:
        ```python
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col

        spark = SparkSession.builder.getOrCreate()

        # Sample data for customers_info DataFrame with missing customer_size_m2 values
        customers_info_data = [
            (1, "customer A", "Location A", "Region A", "Chain A", "Classification A", None, "2023-01-01", 40.123, -75.456),
            (2, "customer B", "Location B", "Region B", "Chain B", "Classification B", None, "2023-02-01", 41.234, -76.567),
            (3, "customer C", "Location C", "Region C", "Chain C", "Classification C", None, "2023-03-01", 42.345, -77.678),
        ]
        customers_info_columns = [
            "customer_id", "customer_name", "customer_location", "region_name", "chain_name",
            "customer_classification", "customer_size_m2", "customer_start_date", "latitude", "longitude"
        ]
        customers_info_df = spark.createDataFrame(customers_info_data, customers_info_columns)

        # Sample data for spine DataFrame
        spine_data = [
            (1, "2023-01-01"), (2, "2023-02-01"), (3, "2023-03-01"),
            (1, "2023-02-01"), (2, "2023-03-01"), (3, "2023-04-01"),
        ]
        spine_columns = ["_id", "_observ_end_dt"]
        spine_df = spark.createDataFrame(spine_data, spine_columns)

        # Create the primary customers table with missing customer size values filled using the median
        prm_customers = create_prm_customers(spine_df, customers_info_df)
        prm_customers.show()
        ```
    """

    # drop duplicates
    customers_cols = set(customers_info.columns) - set(["station_brand"])
    customers_info = customers_info.dropDuplicates(subset=list(customers_cols))

    # create columns to merge
    # customers = create_auxillary_columns(customers_info, unit_of_analysis="customer_id")

    # spine left
    # ftr_customers = spine.join(customers, how="left", on="_id")

    customers_info = customers_info.withColumn(
    #     "mobile",
    #     f.overlay(f.col("mobile"), f.lit("966"), 0, 2)
    # ).withColumn(
        "preferred_language",
        f.when(
            f.upper("nationality").isin(
                ['EGYPTIAN', 'SAUDI', 'EMIRATI', 'KUWAITI', "JORDANIAN", "LEBANESE", "SYRIAN", "IRAQI",
                "SUDANESE", "OMANI", "QATARI", "BAHRAINI", "TUNISIAN","ALGERIAN","MOROCCAN", "LIBYAN",
                "YEMENI", "OTHERS"]
            ),
            f.lit('AR')
        ).when(
            f.col("nationality").isNull(),
            f.lit('AR')
        ).otherwise(f.lit('EN'))
    )

    out = customers_info.select(
        # "_id",
        # "_observ_end_dt",
        "customer_id",
        "age",
        "gender",
        "nationality",
        "mobile",
        "is_owner",
        "is_cash_customer",
        "is_active",
        "preferred_language",
        "station_brand",
    )
    return out.orderBy(["customer_id"])


def create_prm_vehicles(
    vehicles_info: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    The function `create_prm_vehicle` processes vehicle information, cleans data, and returns a
    DataFrame sorted by Customer_id.

    Args:
      vehicles_info (pyspark.sql.DataFrame): The function `create_prm_vehicle` takes a DataFrame `vehicles_info`
    as input and performs the following operations:

    Returns:
      The function `create_prm_vehicle` is returning a modified DataFrame `out` after performing various
    data cleaning and manipulation operations on the input DataFrame `vehicles_info`. The returned
    DataFrame `out` has columns excluding "CreationOn" and "ModifiedOn", and the rows are sorted based
    on the "Customer_id" column.
    """

    vehicles_cols = set(vehicles_info.columns) - set(["station_brand"])
    vehicles_info = vehicles_info.dropDuplicates(subset=list(vehicles_cols))

    vehicles_info = vehicles_info.withColumn(
        "code_vin",
        f.when(
            f.length(f.col("code_vin")) != 17,
            np.nan
        ).otherwise(f.col("code_vin"))
    ).withColumn(
        "code_vin",
        f.when(
            (f.col("code_vin").isNull()) & (f.length(f.col("plate_number")) == 17),
            f.col("plate_number")
        ).otherwise(f.col("code_vin"))
    ).withColumn(
        "code_vin",
        f.when(
            f.length(f.col("code_vin")) < 6,
            np.nan
        ).otherwise(f.col("code_vin"))
    ).withColumn(
        "plate_number",
        f.when(
            f.length(f.col("plate_number")) > 7,
            np.nan
        ).otherwise(f.col("plate_number"))
    )

    vehicles_info = vehicles_info.drop("creation_on", "modified_on")

    out = vehicles_info.select(
        "customer_id",
        "customer_vehicle_id",
        "maker",
        "is_truck",
        "model",
        "vehicle_brand_level",
        "model_year",
        "transmission_type",
        "plate_number",
        "code_vin"
    )

    return out.orderBy("customer_id")


def create_prm_branches(
    branches_info: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """
    The function `create_prm_branch` removes duplicates from a DataFrame based on specified columns and
    sorts the result by the "Customer_id" column.

    Args:
      branches_info (pyspark.sql.DataFrame): The `branches_info` parameter is expected to be a pandas DataFrame
    containing information about branches. The function `create_prm_branch` drops duplicates from the
    DataFrame based on certain columns and then sorts the resulting DataFrame by the "Customer_id"
    column before returning it.

    Returns:
      The function `create_prm_branch` is returning the input DataFrame `branches_info` after dropping
    duplicates based on columns other than "station_brand" and sorting the result by the "Customer_id"
    column.
    """

    # drop duplicates
    branches_cols = set(branches_info.columns) - set(["station_brand"])
    branches_info = branches_info.dropDuplicates(subset=list(branches_cols))

    out = branches_info.select(
        "branch_id",
        "longitude",
        "latitude",
        "branch_code",
        "branch_type",
        "is_active",
        "city",
    )

    return out.orderBy("branch_id")

