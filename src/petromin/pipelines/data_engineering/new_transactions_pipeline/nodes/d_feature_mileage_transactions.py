"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import typing as tp
import pandas as pd
import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window
from feature_generation.v1.nodes.features.create_column import (
    create_columns_from_config,
)

def create_mileage_features(
    spine: pyspark.sql.DataFrame,
    ftr_sales: pyspark.sql.DataFrame,
    ftr_customer_vehicle: pyspark.sql.DataFrame,
    oem_rules: pd.DataFrame,
) -> pyspark.sql.DataFrame:


    ftr_mileage_df = _compute_mileage_features(ftr_sales)
    forecast_mileage_df = _compute_forecast_features(spine, ftr_mileage_df)
    forecast_mileage_product_df = _compute_is_due_features(spine, forecast_mileage_df, ftr_customer_vehicle, oem_rules)
    target_mileage_df = _compute_target_mileage_features(ftr_mileage_df)

    out = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        forecast_mileage_product_df,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        target_mileage_df,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).orderBy(["_id", "_observ_end_dt"])

    return out


def _compute_target_mileage_features(
    ftr_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    win_id_monthly = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt"))

    out = ftr_sales.withColumn(
        "target_mileage_1",
        f.first("month_max_current_mileage", ignorenulls=True).over(win_id_monthly.rowsBetween(1, Window.unboundedFollowing))
    )

    return out.select(
        "_id",
        "_observ_end_dt",
        "target_mileage_1"
    ).orderBy(["_id", "_observ_end_dt"])


def _compute_old_mileage_features(
    base_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    ##### Previous

    win_id = Window.partitionBy("_id").orderBy("transaction_dt").rowsBetween(Window.unboundedPreceding, -1)
    win_id_all = Window.partitionBy("_id").orderBy("transaction_dt").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    win_id_last_3 = Window.partitionBy("_id").orderBy("transaction_dt").rowsBetween(-3, -1)
    win_id_next_3 = Window.partitionBy("_id").orderBy("transaction_dt").rowsBetween(1, 3)

    win_id_monthly = Window.partitionBy("_id").orderBy("_observ_end_dt")

    ftr_trx_mileage = base_sales.groupBy(
        "_id", "_observ_end_dt", "transaction_id", "transaction_dt"
    ).agg(
        f.mean("current_mileage").alias("current_mileage"),
        f.mean("previous_mileage").alias("previous_mileage"),
    ).withColumn(
        "current_mileage",
        f.when(
            (f.abs(f.col("current_mileage") - f.avg("current_mileage").over(win_id_last_3)) > f.std("current_mileage").over(win_id_all)) & (f.avg("current_mileage").over(win_id_last_3).isNotNull()),
            None
        ).when(
            (f.abs(f.col("current_mileage") - f.avg("current_mileage").over(win_id_next_3)) > f.std("current_mileage").over(win_id_all)) & (f.avg("current_mileage").over(win_id_next_3).isNotNull()),
            None
        ).otherwise(f.col("current_mileage"))
    ).withColumn(
        "current_mileage",
        f.when(
            (
                (f.col("current_mileage") - f.last("current_mileage").over(win_id)) / 
                f.when(f.date_diff(f.col("transaction_dt"), f.last("transaction_dt").over(win_id)) == 0, 1).otherwise(f.date_diff(f.col("transaction_dt"), f.last("transaction_dt").over(win_id)))
            ) > 1200,
            None
        ).otherwise(f.col("current_mileage"))
    ).withColumn(
        "transaction_dt",
        f.when(
            f.col("current_mileage").isNull(), None
        ).otherwise(f.col("transaction_dt"))
    ).withColumn(
        "last_mileage",
        f.last("current_mileage", ignorenulls=True).over(win_id)
    ).withColumn(
        "last_transaction_dt",
        f.last("transaction_dt", ignorenulls=True).over(win_id)
    ).withColumn(
        "run_mileage",
        f.col("current_mileage") - f.col("last_mileage")
    ).withColumn(
        "increased_period",
        f.date_diff(f.col("transaction_dt"), f.col("last_transaction_dt"))
    ).withColumn(
        "avg_mileage_per_day",
        f.col("run_mileage") / f.col("increased_period")
    )

    w11 = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt")).rowsBetween(-11, 0)

    ftr_unit_mileage = ftr_trx_mileage.groupBy(
        "_id", "_observ_end_dt",
    ).agg(
        f.last("transaction_dt", ignorenulls=True).alias("last_transaction_dt"),
        f.min("current_mileage").alias("month_min_current_mileage"),
        f.max("current_mileage").alias("month_max_current_mileage"),
        f.mean("run_mileage").alias("month_avg_run_mileage"),
        f.mean("increased_period").alias("month_avg_run_period"),
        f.mean("avg_mileage_per_day").alias("month_avg_mileage_per_day"),
    ).withColumn(
        "month_avg_mileage_per_day",
        f.when(
            f.col("month_avg_mileage_per_day").isNull(),
            f.round(f.first("month_avg_mileage_per_day", ignorenulls=True).over(win_id_monthly.rowsBetween(0, Window.unboundedFollowing)), 2)
        ).otherwise(f.col("month_avg_mileage_per_day"))
    ).withColumn(
        "month_avg_mileage_per_day",
        f.when(
            (f.col("month_min_current_mileage").isNotNull()) &
            ((f.col("month_avg_mileage_per_day") > 1500) | (f.col("month_avg_mileage_per_day").isNull())),
            # global_avg_mileage
            f.lit(55)
        ).otherwise(f.col("month_avg_mileage_per_day"))
    ).withColumn(
        "is_single_visit",
        f.when(
            (f.col("month_max_current_mileage").isNull() |
            f.col("month_min_current_mileage").isNull()),
            None
        ).when(
            f.col("month_max_current_mileage") == f.col("month_min_current_mileage"),
            1
        ).otherwise(0)
    ).withColumn(
        "is_multiple_visit",
        f.when(
            (f.col("month_max_current_mileage").isNull() |
            f.col("month_min_current_mileage").isNull()),
            None
        ).when(
            f.col("month_max_current_mileage") == f.col("month_min_current_mileage"),
            0
        ).otherwise(1)
    ).withColumn(
        "month_avg_mileage_per_day_nullif_past_11_next_0_months",
        f.avg(
            f.replace(f.col("month_avg_mileage_per_day"), f.lit(0.0))
        ).over(w11)
    ).withColumn(
        "estimated_days_to_change_mineral_oil",
        f.when(
            f.col("month_min_current_mileage").isNotNull(),
            f.round(f.lit(5000) / f.col("month_avg_mileage_per_day_nullif_past_11_next_0_months")) # mineral oil
        ).otherwise(None)
    ).withColumn(
        "estimated_days_to_change_synthetic_oil",
        f.when(
            f.col("month_min_current_mileage").isNotNull(),
            f.round(f.lit(10000) / f.col("month_avg_mileage_per_day_nullif_past_11_next_0_months")) # synthetic oil
        ).otherwise(None)
    ).withColumn(
        "estimated_days_to_change_mineral_oil",
        f.last("estimated_days_to_change_mineral_oil", ignorenulls=True).over(win_id_monthly.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "estimated_days_to_change_synthetic_oil",
        f.last("estimated_days_to_change_synthetic_oil", ignorenulls=True).over(win_id_monthly.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
    #     "last_transaction_dt",
    #     f.last("last_transaction_dt", ignorenulls=True).over(win_id_monthly.rangeBetween(Window.unboundedPreceding, 0))
    # ).withColumn(
        "days_since_oil_last_transactions",
        f.date_diff("_observ_end_dt", "last_transaction_dt")
    ).withColumn(
        "days_since_oil_synthetic_last_transactions",
        f.date_diff("_observ_end_dt", "last_transaction_dt")
    ).withColumn(
        "days_until_mineral_oil_change",
        f.col("estimated_days_to_change_mineral_oil") - f.col("days_since_oil_last_transactions")
    ).withColumn(
        "days_until_synthetic_oil_change",
        f.col("estimated_days_to_change_synthetic_oil") - f.col("days_since_oil_synthetic_last_transactions")
    ).withColumn(
        "below_estimated_mineral_oil_change",
        (f.col("days_until_mineral_oil_change") > 0).cast("int")
    ).withColumn(
        "below_estimated_synthetic_oil_change",
        (f.col("days_until_synthetic_oil_change") > 0).cast("int")
    ).select(
        "_id",
        "_observ_end_dt",
        # "last_transaction_dt",
        "month_min_current_mileage",
        "month_max_current_mileage",
        "month_avg_run_mileage",
        "month_avg_run_period",
        "month_avg_mileage_per_day",
        "month_avg_mileage_per_day_nullif_past_11_next_0_months",
        "is_single_visit",
        "is_multiple_visit",
        "estimated_days_to_change_mineral_oil",
        "estimated_days_to_change_synthetic_oil",
        "days_since_oil_last_transactions",
        "days_since_oil_synthetic_last_transactions",
        "days_until_mineral_oil_change",
        "days_until_synthetic_oil_change",
        "below_estimated_mineral_oil_change",
        "below_estimated_synthetic_oil_change",
    )

    out = base_sales.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_unit_mileage,
        on=["_id", "_observ_end_dt"],
        how="left"
    )

    return out.orderBy("_id", "_observ_end_dt")


def _compute_mileage_features(
    base_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    w_id_transactions = Window.partitionBy("_id").orderBy("transaction_dt")

    w_id_past_transactions = Window.partitionBy("_id").orderBy("transaction_dt").rowsBetween(Window.unboundedPreceding, -1)

    mileage_sales = base_sales.groupBy(
        "_id", "_observ_end_dt", "transaction_id", "transaction_dt"
    ).agg(
        f.min("current_mileage").alias("current_mileage"),
    ).filter(
        f.col("transaction_id").isNotNull()
    )

    adjusted_mileage_sales = mileage_sales.withColumn(
        "current_mileage",
        f.when(
            (f.col("current_mileage").isNotNull()) &
            (f.col("current_mileage") < 100),
            None
        ).otherwise(f.col("current_mileage"))
    ).withColumn(
        "current_mileage",
        f.when(
            (f.col("current_mileage").isNotNull()) &
            (f.max("current_mileage").over(Window.partitionBy("_id").orderBy("transaction_dt").rowsBetween(Window.unboundedPreceding, -1)) > 900000) &
            (f.col("current_mileage") < 900000),
            f.col("current_mileage") + 1000000
        ).otherwise(f.col("current_mileage"))
    ).withColumn(
        "current_mileage",
        f.when(
            (f.col("current_mileage").isNotNull()) &
            (f.col("current_mileage") == f.last("current_mileage", ignorenulls=True).over(w_id_past_transactions)) &
            (f.date_diff("transaction_dt", f.last("transaction_dt", ignorenulls=True).over(w_id_past_transactions)) > 0),
            None
        ).otherwise(f.col("current_mileage"))
    ).withColumn(
        "transaction_dt",
        f.when(
            f.col("current_mileage").isNull(),
            None
        ).otherwise(f.col("transaction_dt"))
    ).withColumn(
        "last_transaction_dt",
        f.last("transaction_dt", ignorenulls=True).over(w_id_past_transactions),
    ).withColumn(
        "last_current_mileage",
        f.last("current_mileage", ignorenulls=True).over(w_id_past_transactions),
    ).withColumn(
        "last_2_transaction_dt",
        f.last("transaction_dt", ignorenulls=True).over(w_id_transactions.rowsBetween(Window.unboundedPreceding, -2)),
    ).withColumn(
        "last_2_current_mileage",
        f.last("current_mileage", ignorenulls=True).over(w_id_transactions.rowsBetween(Window.unboundedPreceding, -2)),
    ).withColumn(
        "run_mileage",
        f.col("current_mileage") - f.col("last_current_mileage")
    ).withColumn(
        "run_period",
        f.date_diff("transaction_dt", "last_transaction_dt")
    ).withColumn(
        "mpd",
        f.col("run_mileage") / f.col("run_period")
    ).withColumn(
        "run_mileage_2",
        f.col("current_mileage") - f.col("last_2_current_mileage")
    ).withColumn(
        "run_period_2",
        f.date_diff("transaction_dt", "last_2_transaction_dt")
    ).withColumn(
        "mpd_2",
        f.col("run_mileage_2") / f.col("run_period_2")
    ).withColumn(
        "avg_mpd",
        f.mean("mpd").over(w_id_past_transactions)
    ).withColumn(
        "std_mpd",
        f.std("mpd").over(w_id_past_transactions)
    ).withColumn(
        "delta_mpd",
        f.abs(f.col("mpd") - f.mean("mpd").over(w_id_past_transactions))
    ).withColumn(
        "current_mileage",
        f.when(
            (f.abs(f.col("mpd") - f.mean("mpd").over(w_id_past_transactions)) > (f.std("mpd").over(w_id_past_transactions) * 2)) &
            (f.abs(f.col("mpd_2") - f.mean("mpd").over(w_id_past_transactions)) > (f.std("mpd").over(w_id_past_transactions) * 2)),
            None
        ).otherwise(f.col("current_mileage"))
    ).withColumn(
        "transaction_dt",
        f.when(
            f.col("current_mileage").isNull(),
            None
        ).otherwise(f.col("transaction_dt"))
    ).withColumn(
        "run_mileage",
        f.when(
            f.col("current_mileage").isNull(),
            None
        ).otherwise(f.col("run_mileage"))
    ).withColumn(
        "run_period",
        f.when(
            f.col("current_mileage").isNull(),
            None
        ).otherwise(f.col("run_period"))
    ).withColumn(
        "mileage_per_day",
        f.col("run_mileage") / f.col("run_period")
    )

    client_mileage_df = adjusted_mileage_sales.select(
        "_id",
        "_observ_end_dt",
        "transaction_id",
        "transaction_dt",
        "current_mileage",
        "run_mileage",
        "run_period",
        "mileage_per_day",
    )

    client_product_df = base_sales.select(
        "_id", "_observ_end_dt", "transaction_id", "major_category"
    ).filter(
        f.col("transaction_id").isNotNull()
    ).drop("transaction_id")

    client_product_mileage_df = base_sales.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        client_mileage_df,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        client_product_df,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).orderBy("_id", "_observ_end_dt")

    mileage_pivot_product_view_df = client_product_mileage_df.filter(
        f.col("major_category").isin(f.lit("oil"), f.lit("oil_synthetic"))
    ).groupBy(
        "_id", "_observ_end_dt",
    ).pivot(
        "major_category"
    ).agg(
        f.last("transaction_dt", ignorenulls=True).alias("last_transaction_dt"), 
        f.last("current_mileage", ignorenulls=True).alias("last_mileage"),
    )

    drop_columns = [col for col in mileage_pivot_product_view_df.columns if "null" in col]

    mileage_pivot_product_view_df = mileage_pivot_product_view_df.drop(*drop_columns)

    mileage_pivot_customer_view_df = client_product_mileage_df.groupBy(
        "_id", "_observ_end_dt"
    ).agg(
        f.last("transaction_dt", ignorenulls=True).alias("customer_transaction_dt"),
        f.last("transaction_id", ignorenulls=True).alias("customer_last_monthly_transaction_id"),
        f.last("current_mileage", ignorenulls=True).alias("customer_current_mileage"),
        f.min("current_mileage").alias("month_min_current_mileage"),
        f.max("current_mileage").alias("month_max_current_mileage"),
        f.mean("run_mileage").alias("month_avg_run_mileage"),
        f.mean("run_period").alias("month_avg_run_period"),
        f.mean("mileage_per_day").alias("customer_mileage_per_day"),
    )

    # join product and customer tables
    client_mileage_product_pivot = mileage_pivot_customer_view_df.join(
        mileage_pivot_product_view_df,
        on=["_id", "_observ_end_dt"],
        how="left"
    )

    w_id_observ = Window.partitionBy("_id").orderBy("_observ_end_dt")
    w_id_past_observ = Window.partitionBy("_id").orderBy("_observ_end_dt").rowsBetween(Window.unboundedPreceding, -1)
    w_id_next_observ = Window.partitionBy("_id").orderBy("_observ_end_dt").rowsBetween(1, Window.unboundedFollowing)

    client_mileage_product_pivot = client_mileage_product_pivot.withColumn(
        f"customer_last_current_mileage",
        f.last(f"customer_current_mileage", ignorenulls=True).over(w_id_past_observ)
    ).withColumn(
        f"customer_last_transaction_dt",
        f.last(f"customer_transaction_dt", ignorenulls=True).over(w_id_past_observ)
    ).withColumn(
        f"customer_last_transaction_id",
        f.last(f"customer_last_monthly_transaction_id", ignorenulls=True).over(w_id_observ.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        f"customer_run_mileage",
        f.col(f"customer_current_mileage") - f.col(f"customer_last_current_mileage")
    ).withColumn(
        f"customer_run_period",
        f.date_diff(f"customer_transaction_dt", f"customer_last_transaction_dt")
    ).withColumn(
        f"customer_mileage_per_day",
        f.col(f"customer_run_mileage") / f.col(f"customer_run_period")
    ).withColumn(
        f"customer_mileage_per_day",
        f.when(
            f.col(f"customer_mileage_per_day") < 0,
            None
        ).when(
            f.col(f"customer_mileage_per_day") > 700,
            None
        ).otherwise(f.col(f"customer_mileage_per_day"))
    ).withColumn(
        f"customer_filled_mileage_per_day",
        f.when(
            f.col("customer_mileage_per_day").isNull(),
            f.round(f.first(f"customer_mileage_per_day", ignorenulls=True).over(w_id_next_observ))
        ).otherwise(f.col("customer_mileage_per_day"))
    ).withColumn(
        f"customer_avg_mileage_per_day",
        f.when(
            f.col("customer_filled_mileage_per_day").isNotNull(),
            f.round(f.mean("customer_filled_mileage_per_day").over(w_id_observ.rowsBetween(-11, 0)))
        ).otherwise(None)
    ).withColumn(
        f"customer_avg_mileage_per_day",
        f.when(
            f.col(f"customer_avg_mileage_per_day").isNull(),
            f.round(f.last(f"customer_avg_mileage_per_day", ignorenulls=True).over(w_id_past_observ))
        ).otherwise(f.col(f"customer_avg_mileage_per_day"))
    ).withColumn(
        f"customer_last_mileage_per_day",
        f.last(f"customer_mileage_per_day", ignorenulls=True).over(w_id_observ.rowsBetween(-12, -1))
    ).withColumn(
        f"customer_mileage_per_day",
        f.when(
            f.col(f"customer_mileage_per_day") < 0,
            f.lit(None)
        ).otherwise(f.col(f"customer_mileage_per_day"))
    ).withColumn(
        f"customer_avg_mileage_per_day",
        f.when(
            f.col(f"customer_avg_mileage_per_day").isNull(),
            f.lit(55) # 20000 km in a year
        ).otherwise(f.col(f"customer_avg_mileage_per_day"))
    ).withColumn(
        "is_single_visit",
        f.when(
            (f.col("month_max_current_mileage").isNull() |
            f.col("month_min_current_mileage").isNull()),
            None
        ).when(
            f.col("month_max_current_mileage") == f.col("month_min_current_mileage"),
            1
        ).otherwise(0)
    ).withColumn(
        "is_multiple_visit",
        f.when(
            (f.col("month_max_current_mileage").isNull() |
            f.col("month_min_current_mileage").isNull()),
            None
        ).when(
            f.col("month_max_current_mileage") == f.col("month_min_current_mileage"),
            0
        ).otherwise(1)
    ).withColumn(
        "personal_interval_km",
        f.mean("customer_run_mileage").over(w_id_observ.rowsBetween(-11, 0)) # Rolling 12-month average of their actual km between visits
    ).withColumn(
        "smart_target_mineral",
        f.when(
            f.col("personal_interval_km").isNull(), f.lit(5000) # Fallback for new customers
        ).when(
            f.col("personal_interval_km") > 6000, f.lit(5000) # Cap for "leakers"
        ).otherwise(f.col("personal_interval_km")) # Personalized interval for loyalists
    ).withColumn(
        "smart_target_synthetic",
        f.when(
            f.col("personal_interval_km").isNull(), f.lit(10000)
        ).when(
            f.col("personal_interval_km") > 12000, f.lit(10000)
        ).otherwise(f.col("personal_interval_km"))
    ).withColumn(
        "estimated_days_to_change_mineral_oil",
        f.when(
            f.col("month_min_current_mileage").isNotNull(),
            f.round(f.col("smart_target_mineral") / f.col("customer_avg_mileage_per_day")) # Use smart target instead of lit(5000)
        ).otherwise(None)
    ).withColumn(
        "estimated_days_to_change_synthetic_oil",
        f.when(
            f.col("month_min_current_mileage").isNotNull(),
            f.round(f.col("smart_target_synthetic") / f.col("customer_avg_mileage_per_day")) # Use smart target instead of lit(10000)
        ).otherwise(None)
    ).withColumn(
        "estimated_days_to_change_mineral_oil",
        f.last("estimated_days_to_change_mineral_oil", ignorenulls=True).over(w_id_observ.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "estimated_days_to_change_synthetic_oil",
        f.last("estimated_days_to_change_synthetic_oil", ignorenulls=True).over(w_id_observ.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "days_since_oil_last_transactions",
        f.date_diff("_observ_end_dt", "customer_last_transaction_dt")
    ).withColumn(
        "days_since_oil_synthetic_last_transactions",
        f.date_diff("_observ_end_dt", "customer_last_transaction_dt")
    ).withColumn(
        "days_until_mineral_oil_change",
        f.col("estimated_days_to_change_mineral_oil") - f.col("days_since_oil_last_transactions")
    ).withColumn(
        "days_until_synthetic_oil_change",
        f.col("estimated_days_to_change_synthetic_oil") - f.col("days_since_oil_synthetic_last_transactions")
    ).withColumn(
        "below_estimated_mineral_oil_change",
        (f.col("days_until_mineral_oil_change") > 0).cast("int")
    ).withColumn(
        "below_estimated_synthetic_oil_change",
        (f.col("days_until_synthetic_oil_change") > 0).cast("int")
    )

    out = client_mileage_product_pivot.select(
        '_id',
        '_observ_end_dt',
        "customer_last_transaction_id",
        'customer_current_mileage',
        'month_min_current_mileage',
        'month_max_current_mileage',
        "customer_run_mileage",
        "customer_run_period",
        'month_avg_run_mileage',
        'month_avg_run_period',
        'customer_mileage_per_day',
        'oil_last_mileage',
        'oil_synthetic_last_mileage',
        'customer_last_current_mileage',
        'customer_filled_mileage_per_day',
        'customer_avg_mileage_per_day',
        'customer_last_mileage_per_day',
        'is_single_visit',
        'is_multiple_visit',
        'oil_last_transaction_dt',
        'oil_synthetic_last_transaction_dt',
        'customer_last_transaction_dt',
        'below_estimated_mineral_oil_change',
        'below_estimated_synthetic_oil_change',

    )

    return out.orderBy("_id", "_observ_end_dt")


def _compute_forecast_features(
    spine: pyspark.sql.DataFrame,
    ftr_mileage: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    w_id_observ = Window.partitionBy("_id").orderBy("_observ_end_dt")
    w_id_past_observ = Window.partitionBy("_id").orderBy("_observ_end_dt").rowsBetween(Window.unboundedPreceding, -1)


    client_mileage_product_forecast = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_mileage,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).orderBy("_id", "_observ_end_dt")

    out = client_mileage_product_forecast.withColumn(
        f"customer_days_since_last_trx",
        f.date_diff("_observ_end_dt", f"customer_last_transaction_dt")
    ).withColumn(
        f"customer_mileage_delta",
        f.when(
            f.col("customer_current_mileage").isNull(),
            f.col(f"customer_days_since_last_trx") * f.col(f"customer_avg_mileage_per_day")
        ).otherwise(f.col(f"customer_days_since_last_trx") * f.col(f"customer_avg_mileage_per_day")) # TO BE Reviewed
    ).withColumn(
        f"customer_mileage_original_forecast",
        f.col(f"customer_mileage_delta") + f.col("customer_last_current_mileage")
    ).withColumn(
        f"customer_mileage_forecast",
        f.when(
            f.col("customer_current_mileage").isNull(),
            f.col(f"customer_mileage_delta") + f.col("customer_last_current_mileage")
        ).otherwise(f.col("customer_current_mileage"))
    ).withColumn(
        f"customer_mileage_last_forecast",
        f.lag(f"customer_mileage_forecast").over(w_id_observ)
    ).withColumn(
        f"customer_mileage_last_forecast",
        f.when(
            f.col(f"customer_mileage_last_forecast") > f.col(f"customer_mileage_forecast"),
            None
        ).otherwise(f.col(f"customer_mileage_last_forecast"))
    ).withColumn(
        "oil_last_mileage",
        f.when(
            f.col("oil_last_mileage").isNull(),
            f.last("oil_last_mileage", ignorenulls=True).over(w_id_past_observ)
        ).otherwise(f.col("oil_last_mileage"))
    ).withColumn(
        "oil_synthetic_last_mileage",
        f.when(
            f.col("oil_synthetic_last_mileage").isNull(),
            f.last("oil_synthetic_last_mileage", ignorenulls=True).over(w_id_past_observ)
        ).otherwise(f.col("oil_synthetic_last_mileage"))
    ).withColumn(
        "oil_last_transaction_dt",
        f.when(
            f.col("oil_last_transaction_dt").isNull(),
            f.last("oil_last_transaction_dt", ignorenulls=True).over(w_id_past_observ)
        ).otherwise(f.col("oil_last_transaction_dt"))
    ).withColumn(
        "oil_synthetic_last_transaction_dt",
        f.when(
            f.col("oil_synthetic_last_transaction_dt").isNull(),
            f.last("oil_synthetic_last_transaction_dt", ignorenulls=True).over(w_id_past_observ)
        ).otherwise(f.col("oil_synthetic_last_transaction_dt"))
    ).withColumn(
        "delta_last_mineral_oil_change",
        f.col("customer_mileage_forecast") - f.col("oil_last_mileage")
    ).withColumn(
        "delta_last_synthetic_oil_change",
        f.col("customer_mileage_forecast") - f.col("oil_synthetic_last_mileage")
    ).withColumn(
        "last_delta_last_mineral_oil_change",
        f.lag("delta_last_mineral_oil_change").over(w_id_observ)
    ).withColumn(
        "last_delta_last_synthetic_oil_change",
        f.lag("delta_last_synthetic_oil_change").over(w_id_observ)
    ).withColumn(
        "customer_mineral_oil",
        f.when(
            (f.col("oil_last_transaction_dt").isNotNull()) &
            (f.col("oil_synthetic_last_transaction_dt").isNull()),
            f.lit(1)
        ).when(
            (f.col("oil_last_transaction_dt").isNotNull()) &
            (f.col("oil_last_transaction_dt") > f.col("oil_synthetic_last_transaction_dt")),
            f.lit(1)
        ).otherwise(0)
    ).withColumn(
        "customer_synthetic_oil",
        f.when(
            (f.col("oil_synthetic_last_transaction_dt").isNotNull()) &
            (f.col("oil_last_transaction_dt").isNull()),
            f.lit(1)
        ).when(
            (f.col("oil_synthetic_last_transaction_dt").isNotNull()) &
            (f.col("oil_synthetic_last_transaction_dt") > f.col("oil_last_transaction_dt")),
            f.lit(1)
        ).otherwise(0)
    ).withColumn(
        "floor_last_delta_last_mineral_oil_change",
        f.floor(f.col("last_delta_last_mineral_oil_change") / f.lit(5000))
    ).withColumn(
        "floor_delta_last_mineral_oil_change",
        f.floor(f.col("delta_last_mineral_oil_change") / f.lit(5000))
    ).withColumn(
        "floor_last_delta_last_synthetic_oil_change",
        f.floor(f.col("last_delta_last_synthetic_oil_change") / f.lit(10000))
    ).withColumn(
        "floor_delta_last_synthetic_oil_change",
        f.floor(f.col("delta_last_synthetic_oil_change") / f.lit(10000))
    ).withColumn(
        "is_due_mineral_oil",
        f.when(
            (f.col("customer_mineral_oil") == f.lit(1)) &
            (f.col("floor_last_delta_last_mineral_oil_change") < f.col("floor_delta_last_mineral_oil_change")),
            f.lit(1)
        ).otherwise(f.lit(0))
    ).withColumn(
        "is_due_synthetic_oil",
        f.when(
            (f.col("customer_synthetic_oil") == f.lit(1)) &
            (f.col("floor_last_delta_last_synthetic_oil_change") < f.col("floor_delta_last_synthetic_oil_change")),
            f.lit(1)
        ).otherwise(f.lit(0))
    )

    return out.orderBy("_id", "_observ_end_dt")


def _compute_is_due_features(
    spine: pyspark.sql.DataFrame,
    ftr_mileage_forecast: pyspark.sql.DataFrame,
    ftr_customer_vehicle: pyspark.sql.DataFrame,
    oem_rules: pd.DataFrame,
) -> pyspark.sql.DataFrame:

    oem_rules["car model"] = oem_rules["car model"].replace("All", None)
    oem_rules["mileages"] = oem_rules["mileages"].str.split(",").str.join("").astype(int)

    bu_dict = oem_rules.set_index("product").to_dict()["servicing BU"]
    product_list = oem_rules["product"].unique()

    ftr_mileage_product = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_mileage_forecast,
        on=["_id", "_observ_end_dt"],
        how="left",
    ).join(
        ftr_customer_vehicle.select(
            "_id", "_observ_end_dt",
            "model",
            "maker",
        ),
        on=["_id", "_observ_end_dt"],
        how="left",
    )

    def _update_filter(product):
        filter_product = oem_rules["product"] == product
        filter_all_cars = oem_rules["car model"].isna()

        filter_model = filter_product & (~filter_all_cars)
        filter_brand = filter_product & filter_all_cars

        return filter_model, filter_brand

    for product in product_list:

        if product in ["Mineral Oil", "Synthetic Oil"]:
            continue
        filter_model, filter_brand = _update_filter(product)
        model_mileage_dict = oem_rules[filter_model].set_index("car model").to_dict()["mileages"]
        brand_mileage_dict = oem_rules[filter_brand].set_index("car brand").to_dict()["mileages"]

        ftr_mileage_product = ftr_mileage_product.withColumn(
                f"is_due_{product.replace(" ", "_").lower()}",
                f.lit(0)
            )

        for model, mileage in model_mileage_dict.items():
            ftr_mileage_product = ftr_mileage_product.withColumn(
                f"is_due_{product.replace(" ", "_").lower()}",
                f.when(
                    (f.col("model") == f.lit(model.upper())) &
                    (f.floor(f.col("customer_mileage_last_forecast") / mileage) < f.floor(f.col("customer_mileage_forecast") / mileage)),
                    1
                ).otherwise(f.col(f"is_due_{product.replace(" ", "_").lower()}"))
            )

        for brand, mileage in brand_mileage_dict.items():
            ftr_mileage_product = ftr_mileage_product.withColumn(
                f"is_due_{product.replace(" ", "_").lower()}",
                f.when(
                    (f.col(f"is_due_{product.replace(" ", "_").lower()}") < 1) &
                    (f.col("maker") == f.lit(brand.upper())) &
                    (f.floor(f.col("customer_mileage_last_forecast") / mileage) < f.floor(f.col("customer_mileage_forecast") / mileage)),
                    1
                ).when(
                    (f.col(f"is_due_{product.replace(" ", "_").lower()}") < 1) &
                    (f.lit(brand.upper()).alias("brand") == f.lit("ALL").alias("brand_all")) &
                    (f.floor(f.col("customer_mileage_last_forecast") / mileage) < f.floor(f.col("customer_mileage_forecast") / mileage)),
                    1
                ).otherwise(f.col(f"is_due_{product.replace(" ", "_").lower()}"))
            )

    for product in product_list:
        product_TP_val = oem_rules.set_index("product").to_dict()["TP"][product]
        ftr_mileage_product = ftr_mileage_product.withColumn(
                f"expected_revenue_{product.replace(" ", "_").lower()}",
                f.col(f"is_due_{product.replace(" ", "_").lower()}") * f.lit(product_TP_val)
            )

    ftr_mileage_product = ftr_mileage_product.withColumn(
        "is_due_PE",
        f.lit(0)
    ).withColumn(
        "is_due_PAC",
        f.lit(0)
    ).withColumn(
        "expected_revenue_PE",
        f.lit(0)
    ).withColumn(
        "expected_revenue_PAC",
        f.lit(0)
    )

    for product, bu in bu_dict.items():
        ftr_mileage_product = ftr_mileage_product.withColumn(
            f"is_due_{bu}",
            f.when(
                f.col(f"is_due_{product.replace(" ", "_").lower()}") > 0,
                1
            ).otherwise(f.col(f"is_due_{bu}"))
        ).withColumn(
            f"expected_revenue_{bu}",
            f.when(
                f.col(f"is_due_{product.replace(" ", "_").lower()}") > 0,
                f.col(f"expected_revenue_{bu}") + f.col(f"expected_revenue_{product.replace(" ", "_").lower()}")
            ).otherwise(f.col(f"expected_revenue_{bu}"))
        )

    ftr_mileage_product = ftr_mileage_product.withColumn(
        "is_due_PE",
        f.when(
            f.col(f"is_due_PAC") > 0,
            0
        ).otherwise(f.col("is_due_PE"))
    ).withColumn(
        "expected_revenue_PAC",
        f.when(
            f.col(f"expected_revenue_PAC") > 0,
            f.col(f"expected_revenue_PAC") + f.col(f"expected_revenue_PE")
        ).otherwise(f.col("expected_revenue_PAC"))
    ).withColumn(
        "expected_revenue_PE",
        f.when(
            f.col(f"expected_revenue_PAC") > 0,
            0
        ).otherwise(f.col("expected_revenue_PE"))
    ).withColumn(
        "monthly_expected_revenue",
        f.expr(" + ".join([f"expected_revenue_{product.replace(" ", "_").lower()}" for product in product_list]))
    )

    out = ftr_mileage_product.drop(
        "model",
        "maker",
    )

    return out.orderBy("_id", "_observ_end_dt")
