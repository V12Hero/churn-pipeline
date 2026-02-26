"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import logging
import typing as tp

import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window

logger = logging.getLogger(__name__)


def create_sales_features(
    spine: pyspark.sql.DataFrame,
    base_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    
    ftr_trx_sales = _compute_global_sales_features(base_sales)

    logger.info(
        f"transactional sales features:\t\t\t\t({ftr_trx_sales.count()}, {len(ftr_trx_sales.columns)})",
    )

    ftr_trx_sales_time = _compute_time_of_sales_features(base_sales)

    logger.info(
        f"transactional sales time features:\t\t\t({ftr_trx_sales_time.count()}, {len(ftr_trx_sales_time.columns)})",
    )

    ftr_trx_sales_category = _compute_category_sales_features(base_sales)

    logger.info(
        f"transactional category features:\t\t\t({ftr_trx_sales_category.count()}, {len(ftr_trx_sales_category.columns)})",
    )

    ftr_last_trx_sales = _compute_last_sales_features(base_sales)

    logger.info(
        f"transactional last trx features:\t\t\t({ftr_last_trx_sales.count()}, {len(ftr_last_trx_sales.columns)})",
    )

    # merge dataframes
    ftr_sales = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_trx_sales,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_trx_sales_category,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_trx_sales_time,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_last_trx_sales,
        on=["_id", "_observ_end_dt"],
        how="left"
    )

    share_columns = list(ftr_trx_sales_category.columns)
    ftr_sales_w_shares = _compute_category_shares_features(ftr_sales, share_columns)

    logger.info(
        f"transactional ftr sales data:\t\t\t\t({ftr_sales_w_shares.count()}, {len(ftr_sales_w_shares.columns)})",
    )

    out = ftr_sales_w_shares.orderBy(["_id", "_observ_end_dt"])

    return out


def _compute_global_sales_features(
    base_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    # Calculate aggregated features for sale
    ftr_trx_sales = base_sales.withColumn(
        "discount_percent",
        f.col("discount_amount") / (f.col("sales_amount_net") + f.col("discount_amount"))
    ).groupBy(["_id", "_observ_end_dt"]).agg(
        f.sum("sales_amount").alias("month_total_sales"),
        f.sum("sales_amount_net").alias("month_net_sales"),
        f.sum("total_profit").alias("month_total_profit"),
        f.sum("net_cost").alias("month_total_cost"),
        f.sum("discount_amount").alias("month_total_amount_discounts"),
        f.sum("has_discount").alias("month_total_qty_discounts"),
        f.mean("discount_percent").alias("month_avg_percent_discount"),
        f.countDistinct("product_id").alias("month_distinct_skus_sold"),
        f.countDistinct("branch_id").alias("month_distinct_branches"),
        f.countDistinct("product_category").alias("month_distinct_product_categories"),
        f.countDistinct("transaction_id").alias("month_distinct_transactions"),
    )

    # Compute average ticket, average discount and average sku's per order
    ftr_trx_sales = ftr_trx_sales.withColumn(
        "month_total_discount_percentual",
        f.col("month_total_amount_discounts") / (f.col("month_net_sales") + f.col("month_total_amount_discounts"))
    ).withColumn(
        "month_avg_order",
        f.col("month_net_sales") / f.col("month_distinct_transactions")
    ).withColumn(
        "month_avg_skus_per_order",
        f.col("month_distinct_skus_sold") / f.col("month_distinct_transactions"),
    ).fillna(0)

    out = ftr_trx_sales.orderBy("_id", "_observ_end_dt")

    return out


def _compute_last_sales_features(
    base_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    # Calculate aggregated features for sale
    ftr_last_trx_sales = base_sales.groupBy(
        ["_id", "_observ_end_dt", "transaction_id", "transaction_dt"]
    ).agg(
        f.sum("sales_amount").alias("last_trx_total_sales"),
        f.sum("sales_amount_net").alias("last_trx_net_sales"),
        f.sum("total_profit").alias("last_trx_total_profit"),
        f.sum("net_cost").alias("last_trx_total_cost"),
        f.sum("discount_amount").alias("last_trx_total_amount_discounts"),
        f.sum("has_discount").alias("last_trx_total_qty_discounts"),
        f.avg("current_mileage").alias("last_trx_current_mileage"),
        f.countDistinct("product_id").alias("last_trx_distinct_skus_sold"),
        f.countDistinct("branch_id").alias("last_trx_distinct_branches"),
        f.countDistinct("product_category").alias("last_trx_distinct_product_categories"),
        f.countDistinct("transaction_id").alias("last_trx_distinct_transactions"),
        # TODO: test on notebook
        # f.max(f.when(f.countDistinct("product_category") > 1, 1).otherwise(0)).alias("is_bundle"),
        f.max(f.when(f.lower("product_category").isin("oil"), 1).otherwise(0)).alias("has_oil"),
        f.max(f.when(f.lower("product_category").isin("oil synthetic"), 1).otherwise(0)).alias("has_oil_synthetic"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("oil filter")), 1).otherwise(0)).alias("has_oil_filter"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("air filter")), 1).otherwise(0)).alias("has_air_filter"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("ac ")), 1).otherwise(0)).alias("has_ac"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("tyres")), 1).otherwise(0)).alias("has_tires"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("batt")), 1).otherwise(0)).alias("has_batteries"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("transmission")), 1).otherwise(0)).alias("has_transmission"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("engine")), 1).otherwise(0)).alias("has_engine"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("additives")), 1).otherwise(0)).alias("has_additives"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("plug")), 1).otherwise(0)).alias("has_plug"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("fuel")), 1).otherwise(0)).alias("has_fuel"),
        f.max(f.when(f.contains(f.lower("product_category"), f.lit("gear")), 1).otherwise(0)).alias("has_gear"),
    ).withColumn(
        "last_trx_total_discount_percentual",
        f.col("last_trx_total_amount_discounts") / f.col("last_trx_net_sales"),
    ).withColumn(
        "last_transaction_n_skus",
        f.col("last_trx_distinct_skus_sold") / f.col("last_trx_distinct_transactions"),
    ).withColumnRenamed(
        "transaction_dt", "last_transaction_dt"
    ).fillna(0)

    w1 = Window.partitionBy("_id", "_observ_end_dt").orderBy(f.col("last_transaction_dt")).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    ftr_last_trx_sales_filtered = ftr_last_trx_sales.withColumn(
        "last_transaction_id",
        f.last("transaction_id").over(w1)
    ).filter(
        f.col("transaction_id") == f.col("last_transaction_id")
    )

    out = base_sales.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_last_trx_sales_filtered,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).fillna(0).withColumn(
        "is_active",
        f.when(f.col("last_trx_total_sales") > 0, 1).otherwise(0)
    ).orderBy("_id", "_observ_end_dt")

    return out


def _compute_category_sales_features(
    base_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """Compute category-based sales transactional features.

    This function calculates transactional features for each product category within
    the base sales data.

    Args:
        base_sales (pyspark.sql.DataFrame): DataFrame containing base sales data.

    Returns:
        pyspark.sql.DataFrame: DataFrame with computed category-based sales transactional features.

    Examples:
        Usage example of the _compute_category_sales_transactional_features function:

        ```python
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        # Sample data for base_sales DataFrame
        base_sales_data = [
            (1, '2023-01-01', 'Category A', 100, 90, 10, 5, 'Regional A', 1),
            (1, '2023-01-02', 'Category A', 120, 110, 10, 5, 'Regional A', 2),
            (2, '2023-01-01', 'Category B', 80, 75, 5, 3, 'Regional B', 3),
            (2, '2023-01-02', 'Category B', 70, 65, 5, 3, 'Regional B', 4),
        ]
        base_sales_columns = [
            '_id', '_observ_end_dt', 'product_category', 'sales_amount', 'sales_amount_net',
            'total_profit', 'net_cost', 'regional', 'transaction_id'
        ]
        base_sales_df = spark.createDataFrame(base_sales_data, base_sales_columns)

        # Compute category-based sales transactional features
        category_sales_features = _compute_category_sales_transactional_features(base_sales_df)
        category_sales_features.show()
        ```

    The resulting DataFrame will contain computed transactional features for each
    product category separately.
    """
    base_sales_category = base_sales.withColumn(
        "discount_percent",
        f.col("discount_amount") / (f.col("sales_amount_net") + f.col("discount_amount"))
    )

    ftr_trx_sales_category = (
        base_sales_category.groupBy(["_id", "_observ_end_dt"])
        .pivot("major_category")
        .agg(
            # f.sum("sales_amount").alias("month_total_sales"),
            f.sum("sales_amount_net").alias("month_net_sales"),
            f.sum("total_profit").alias("month_total_profit"),
            f.sum("net_cost").alias("month_total_cost"),
            f.sum("discount_amount").alias("month_total_amount_discounts"),
            f.sum("has_discount").alias("month_total_qty_discounts"),
            f.mean("discount_percent").alias("month_avg_percent_discount"),
        )
    ).fillna(0)

    columns = ["_id", "_observ_end_dt"]

    for col in ftr_trx_sales_category.columns:
        if col not in ["_id", "_observ_end_dt"]:
            new_name = (
                ("month" + col.split("month")[1] + "_category_" + col.split("month")[0])[:-1]
                .lower()
                .replace(".", "")
                .replace(" ", "_")
                .replace("-", "_exi")
            )

            ftr_trx_sales_category = ftr_trx_sales_category.withColumnRenamed(
                col,
                new_name,
            )

            if "null" not in col:
                columns.append(new_name)

    ftr_trx_sales_category = ftr_trx_sales_category.select(columns)

    return ftr_trx_sales_category


def _compute_global_returns_features(
    base_returns: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    """Compute global returns transactional features.

    This function calculates transactional features related to returns within the base returns data.

    Args:
        base_returns (pyspark.sql.DataFrame): DataFrame containing base returns data.

    Returns:
        pyspark.sql.DataFrame: DataFrame with computed global returns transactional features.

    Examples:
        Usage example of the _compute_global_returns_transactional_features function:

        ```python
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        # Sample data for base_returns DataFrame
        base_returns_data = [
            (1, '2023-01-01', 50, 5),
            (1, '2023-01-02', 30, 3),
            (2, '2023-01-01', 20, 2),
            (2, '2023-01-02', 15, 1),
        ]
        base_returns_columns = [
            '_id', '_observ_end_dt', 'sales_amount', 'net_cost'
        ]
        base_returns_df = spark.createDataFrame(base_returns_data, base_returns_columns)

        # Compute global returns transactional features
        global_returns_features = _compute_global_returns_transactional_features(base_returns_df)
        global_returns_features.show()
        ```

    The resulting DataFrame will contain computed transactional features related to returns.
    """
    ftr_trx_returns = (
        base_returns.groupBy(["_id", "_observ_end_dt"])
        .agg(
            f.sum(f.abs("sales_amount")).alias("trx_total_returns"),
            f.sum("net_cost").alias("trx_total_cost_returns"),
        )
        .fillna(0)
    )
    return ftr_trx_returns


def _compute_category_shares_features(
    merged: pyspark.sql.DataFrame, share_columns: tp.List[str]
) -> pyspark.sql.DataFrame:
    """Compute category shares of transactional features.

    This function calculates the share of each category-based transactional feature
    in relation to the total net sales.

    Args:
        merged (pyspark.sql.DataFrame): DataFrame containing merged data
            with transactional features.
        share_columns (List[str]): List of columns representing category-based
            transactional features.

    Returns:
        pyspark.sql.DataFrame: DataFrame with computed category shares of transactional features.

    Examples:
        Usage example of the _compute_category_shares_transactional_features function:

        ```python
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        # Sample data for merged DataFrame
        merged_data = [
            (1, '2023-01-01', 200, 100, 50, 50, 300, 200, 100),
            (1, '2023-01-02', 220, 110, 55, 55, 330, 220, 110),
            (2, '2023-01-01', 100, 50, 25, 25, 150, 100, 50),
            (2, '2023-01-02', 85, 42, 21, 21, 127, 85, 42),
        ]
        merged_columns = [
            '_id', '_observ_end_dt', 'trx_net_sales_category_A', 'trx_total_profit_category_A',
            'trx_total_cost_category_A', 'trx_net_sales_category_B', 'trx_total_profit_category_B',
            'trx_total_cost_category_B',
        ]
        merged_df = spark.createDataFrame(merged_data, merged_columns)

        # List of columns representing category-based transactional features
        category_share_columns = [
            'trx_net_sales_category_A', 'trx_total_profit_category_A', 'trx_total_cost_category_A',
            'trx_net_sales_category_B', 'trx_total_profit_category_B', 'trx_total_cost_category_B',
        ]

        # Compute category shares of transactional features
        category_shares = _compute_category_shares_transactional_features(merged_df, category_share_columns)
        category_shares.show()
        ```

    The resulting DataFrame will contain computed category shares of transactional features
    in relation to the total net sales.
    """

    for col in share_columns:
        if ("net_sales" in col) & (col not in ["_id", "_observ_end_dt"]):

            new_name = col.split("category")[0] + "share" + col.split("category")[1]

            merged = merged.withColumn(new_name, f.col(col) / f.col("month_net_sales"))
 
    return merged.orderBy("_id", "_observ_end_dt")


def _compute_time_of_sales_features(
    base_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    ftr_trx_sales_time = base_sales.filter(
        f.col("transaction_id").isNotNull()
    ).dropDuplicates(
        subset=["_id", "_observ_end_dt", "transaction_id", "transaction_dt"]
    ).withColumn(
        "dayofweek",
        f.dayofweek("transaction_dt")
    ).withColumn(
        "dayofmonth",
        f.dayofmonth("transaction_dt")
    ).withColumn(
        "is_weekday",
        f.when(f.col("dayofweek") < 6, 1).otherwise(0)
    ).withColumn(
        "is_weekend",
        f.when(f.col("dayofweek") > 5, 1).otherwise(0)
    ).withColumn(
        "week_of_month",
        f.date_format("transaction_dt", "W")
    ).withColumn(
        "hourofday",
        f.hour("transaction_dt")
    ).withColumn(
        "is_morning",
        f.when(
            (f.col("hourofday") > 5) & (f.col("hourofday") < 12),
            1
        ).otherwise(0)
    ).withColumn(
        "is_afternoon",
        f.when(
            (f.col("hourofday") > 11) & (f.col("hourofday") < 18),
            1
        ).otherwise(0)
    ).withColumn(
        "is_night",
        f.when(
            (f.col("hourofday") > 17) & (f.col("hourofday") < 24),
            1
        ).otherwise(0)
    ).withColumn(
        "is_afternight",
        f.when(
            (f.col("hourofday") > -1) & (f.col("hourofday") < 5),
            1
        ).otherwise(0)
    ).select(
        "_id",
        "_observ_end_dt",
        "transaction_id",
        "transaction_dt",
        "dayofweek",
        "dayofmonth",
        "is_weekday",
        "is_weekend",
        "week_of_month",
        "hourofday",
        "is_morning",
        "is_afternoon",
        "is_night",
        "is_afternight",
    )

    ftr_trx_sales_time_week = ftr_trx_sales_time.groupBy(
        ["_id", "_observ_end_dt", "transaction_id"]
    ).pivot(
        "week_of_month",
    ).agg(
        f.countDistinct("transaction_id").alias("trx_distinct_transactions"),
    ).fillna(0)

    ftr_trx_sales_time_week = ftr_trx_sales_time_week.withColumn(
        "5",
        f.col("5") + f.col("6")
    )

    columns = ["_id", "_observ_end_dt", "transaction_id"]

    for col in ["1", "2", "3", "4", "5",]:
        if col not in ["_id", "_observ_end_dt", "transaction_id"]:
            new_name = f"trx_distinct_transactions_week_{col}"

            ftr_trx_sales_time_week = ftr_trx_sales_time_week.withColumnRenamed(
                col,
                new_name,
            )

            if "null" not in col:
                columns.append(new_name)

    ftr_trx_sales_time_week = ftr_trx_sales_time_week.select(columns)

    ftr_trx_sales_time = ftr_trx_sales_time.join(
        ftr_trx_sales_time_week,
        on=["_id", "_observ_end_dt", "transaction_id"],
        how="left"
    )

    ftr_trx_sales_time_agg = ftr_trx_sales_time.groupBy(
        ["_id", "_observ_end_dt"]
    ).agg(
        f.avg('dayofweek').alias("avg_day_of_week"),
        f.mode('dayofweek').alias("most_frequent_day_of_week"),
        f.avg('dayofmonth').alias("avg_day_of_month"),
        f.mode('dayofmonth').alias("most_frequent_day_of_month"),
        f.sum('is_weekday').alias("total_transactions_at_weekday"),
        f.sum('is_weekend').alias("total_transactions_at_weekend"),
        f.avg('week_of_month').alias("avg_week_transaction"),
        f.mode('week_of_month').alias("most_frequent_week_transaction"),
        f.avg('hourofday').alias("avg_hour_transaction"),
        f.mode('hourofday').alias("most_frequent_hour_transaction"),
        f.sum('is_morning').alias("total_transactions_at_morning"),
        f.sum('is_afternoon').alias("total_transactions_at_afternoon"),
        f.sum('is_night').alias("total_transactions_at_night"),
        f.sum('is_afternight').alias("total_transactions_at_afternight"),
        f.sum('trx_distinct_transactions_week_1').alias("total_transactions_week_1"),
        f.sum('trx_distinct_transactions_week_2').alias("total_transactions_week_2"),
        f.sum('trx_distinct_transactions_week_3').alias("total_transactions_week_3"),
        f.sum('trx_distinct_transactions_week_4').alias("total_transactions_week_4"),
        f.sum('trx_distinct_transactions_week_5').alias("total_transactions_week_5"),
        # f.sum('trx_distinct_transactions_week_6').alias("total_transactions_week_6")
    )

    out = base_sales.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_trx_sales_time_agg,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).fillna(0).orderBy("_id", "_observ_end_dt")

    return out
