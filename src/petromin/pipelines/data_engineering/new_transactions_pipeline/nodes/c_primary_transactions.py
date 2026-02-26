"""Primary layer nodes."""

import logging
import typing as tp
from datetime import date
from math import radians
from typing import List

import pandas as pd
import pyspark.sql.dataframe
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql import SparkSession
from sklearn.neighbors import BallTree


from .c_primary_spine import create_auxillary_columns

logger = logging.getLogger(__name__)

# TODO:
#   * Add features about time since first transaction
#   * Add features about avg time between transactions
#   * Create flag if current time is normal given client historical Z >= 1.96


def create_prm_transactions(
    transactions_data: pyspark.sql.DataFrame,
    spine: pyspark.sql.DataFrame,
    prm_customers: pyspark.sql.DataFrame,
    prm_vehicles: pyspark.sql.DataFrame,
    prm_geolocation: pyspark.sql.DataFrame,
    branches_to_drop: List,
    products_to_drop: List,
):
    """Create the primary transactions table.

    Args:
        transactions_data (pyspark.sql.DataFrame): DataFrame containing transactional data.
        spine (pyspark.sql.DataFrame): DataFrame representing the spine.
        branches_to_drop (List): List of regionals to drop from transactional data.
        products_to_drop (List): List of product IDs to drop from transactional data.

    Returns:
        pyspark.sql.DataFrame: The primary transactions table.

    Examples:
        Usage examples of the create_prm_transactions function:

        1. Create a primary transactions table with filtering and feature computation:
        ```python
        from pyspark.sql import SparkSession
        from datetime import date

        spark = SparkSession.builder.getOrCreate()

        # Sample data for transactions_data DataFrame
        transactions_data = spark.createDataFrame(
            [(1, '2023-01-01', 'Regional A', 101, 5),
            (1, '2023-01-02', 'Regional A', 102, 3),
            (2, '2023-01-01', 'Regional B', 103, 2)],
            ['customer_id', 'transaction_dt', 'regional', 'product_id', 'quantity'],
            )

        # Sample data for spine DataFrame
        spine_data = [(1, '2023-01-01'), (2, '2023-01-01')]
        spine_columns = ['_id', '_observ_end_dt']
        spine_df = spark.createDataFrame(spine_data, spine_columns)

        # Lists of regionals and products to drop
        branches_to_drop = ['Regional A']
        products_to_drop = [102]

        # Create the primary transactions table
        prm_transactions = create_prm_transactions(
            transactions_data,
            spine_df,
            branches_to_drop,
            products_to_drop)
        prm_transactions.show()
        ```

        2. Create a primary transactions table with no filtering:
        ```python
        from pyspark.sql import SparkSession
        from datetime import date

        spark = SparkSession.builder.getOrCreate()

        # Sample data for transactions_data DataFrame
        transactions_data = spark.createDataFrame([(1, '2023-01-01', 'Regional A', 101, 5),
                                                   (1, '2023-01-02', 'Regional A', 102, 3),
                                                   (2, '2023-01-01', 'Regional B', 103, 2)],
                                                  ['customer_id', 'transaction_dt', 'regional', 'product_id', 'quantity'])

        # Sample data for spine DataFrame
        spine_data = [(1, '2023-01-01'), (2, '2023-01-01')]
        spine_columns = ['_id', '_observ_end_dt']
        spine_df = spark.createDataFrame(spine_data, spine_columns)

        # Empty lists for regionals and products to drop
        branches_to_drop = []
        products_to_drop = []

        # Create the primary transactions table with no filtering
        prm_transactions = create_prm_transactions(transactions_data, spine_df, branches_to_drop, products_to_drop)
        prm_transactions.show()
        ```
    """

    # add transactional ids
    transactions_data = create_auxillary_columns(
        transactions_data, unit_of_analysis=["customer_id", "customer_vehicle_id"], time_column="transaction_dt"
    ).withColumnRenamed("_id", "_key")

    logger.info(
        f"transactional data -\t\t\tintial shape:\t\t\t({transactions_data.count()}, {len(transactions_data.columns)})"
    )

    # filter only transactional data needed
    if branches_to_drop is not None:
        transactions_data = transactions_data.filter(
            ~f.col("branch_id").isin(branches_to_drop)
        )

    logger.info(
        f"transactional data -\t\t\regional filtered: ({transactions_data.count()}, {len(transactions_data.columns)})",
    )

    if branches_to_drop is not None:
        transactions_data = transactions_data.filter(
            ~f.col("product_id").isin(products_to_drop)
        )

    logger.info(
        f"transactional data -\t\t\tproduct_id filtered:\t\t\t({transactions_data.count()}, {len(transactions_data.columns)})",
    )

    # transactional data includes returned products
    returns = transactions_data.filter((f.col("quantity") < 0))
    logger.info(f"returned transactions -\t\t\tintial shape:\t\t\t({returns.count()}, {len(returns.columns)})")

    base_sales = spine.join(
        transactions_data,
        on=["_key", "_observ_end_dt"],
        how="left"
    )

    base_sales = base_sales.withColumn(
        "major_category",
        f.when(
            f.lower("product_category").isin("oil"),
            "oil"
        ).when(
            f.lower("product_category").isin("oil synthetic"),
            "oil_synthetic"
        ).when(
            f.contains(f.lower("product_category"), f.lit("oil filter")),
            "oil_filter"
        ).when(
            f.contains(f.lower("product_category"), f.lit("air filter")),
            "air_filter"
        ).when(
            f.contains(f.lower("product_category"), f.lit("ac ")),
            "ac"
        ).when(
            f.contains(f.lower("product_category"), f.lit("tyres")),
            "tyres"
        ).when(
            f.contains(f.lower("product_category"), f.lit("batt")),
            "batteries"
        ).when(
            f.contains(f.lower("product_category"), f.lit("transmission")),
            "transmission"
        ).when(
            f.contains(f.lower("product_category"), f.lit("engine")),
            "engine"
        ).when(
            f.contains(f.lower("product_category"), f.lit("additives")),
            "additives"
        ).when(
            f.contains(f.lower("product_category"), f.lit("plug")),
            "plug"
        ).when(
            f.contains(f.lower("product_category"), f.lit("fuel")),
            "fuel"
        ).when(
            f.contains(f.lower("product_category"), f.lit("gear")),
            "gear"
        ).when(
            f.contains(f.lower("product_category"), f.lit("brake")),
            "brake"
        ).when(
            f.contains(f.lower("product_category"), f.lit("coolant")),
            "coolant"
        ).otherwise("others")
    )

    # base_returns = spine.join(
    #     returns,
    #     on=["_id", "_observ_end_dt"],
    #     how="left"
    # )

    logger.info(
        f"transactional merge -\t\t\tspine input:\t\t\t({spine.count()}, {len(spine.columns)})",
    )
    logger.info(
        f"transactional merge -\t\t\tspine distinct input:\t\t\t({spine.select('_id', '_observ_end_dt').distinct().count()}, {len(spine.columns)})",
    )
    logger.info(
        f"transactional merge -\t\t\ttransactions input:\t\t\t({transactions_data.count()}, {len(transactions_data.columns)})",
    )
    logger.info(
        f"transactional merge -\t\t\tbase_sales output:\t\t\t({base_sales.count()}, {len(base_sales.columns)})",
    )

    out = base_sales.orderBy(["_id", "_observ_end_dt"])

    logger.info(
        f"transactional out data: ({out.count()}, {len(out.columns)})",
    )

    return out
