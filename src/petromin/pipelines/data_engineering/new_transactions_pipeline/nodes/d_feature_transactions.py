"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import typing as tp

import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window
from feature_generation.v1.nodes.features.create_column import (
    create_columns_from_config,
)


def create_ftr_transactions(
    input_data: pyspark.sql.DataFrame, instructions: tp.Dict, sequential: tp.Dict
) -> pyspark.sql.DataFrame:
    """Return transaction features."""

    features = create_columns_from_config(input_data, instructions, sequential)

    return features.orderBy(["_id", "_observ_end_dt"])


def create_ftr_geolocation(
    input_data: pyspark.sql.DataFrame, instructions: tp.Dict, sequential: tp.Dict
) -> pyspark.sql.DataFrame:
    """Return transaction features."""

    features = create_columns_from_config(input_data, instructions, sequential)

    return features.orderBy(["_id", "_observ_end_dt"])


def compute_churn_features(
    ftr_sales: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    for cat in ["customer", "oil", "oil_synthetic",]:
        ftr_sales = ftr_sales.withColumn(
            f"{cat}_days_since_last_trx",
            f.date_diff("_observ_end_dt", f"{cat}_last_transaction_dt")
        ).withColumn(
            f"{cat}_avg_mileage_per_day",
            f.when(
                (f.col(f"{cat}_avg_mileage_per_day") > 1500) | (f.col(f"{cat}_avg_mileage_per_day").isNull()),
                f.lit(55) # 25000 km in a year
            ).otherwise(f.col(f"{cat}_avg_mileage_per_day"))
        ).withColumn(
            f"{cat}_last_mileage_per_day",
            f.when(
                (f.col(f"{cat}_last_mileage_per_day") > 1500) | (f.col(f"{cat}_last_mileage_per_day").isNull()),
                f.lit(55) # 25000 km in a year
            ).otherwise(f.col(f"{cat}_last_mileage_per_day"))
        ).withColumn(
            f"{cat}_days_until_change",
            f.when(
                f.lit(cat == "oil"),
                f.floor(f.lit(5000) / f.col(f"{cat}_avg_mileage_per_day"))
            ).when(
                f.lit(cat == "oil_synthetic"),
                f.floor(f.lit(10000) / f.col(f"{cat}_avg_mileage_per_day"))
            ).otherwise(None)
        ).withColumn(
            f"{cat}_forecast_mileage",
            f.when(
                f.col("customer_current_mileage").isNull(),
                f.col(f"customer_last_current_mileage") + f.col(f"{cat}_days_since_last_trx") * f.col(f"{cat}_avg_mileage_per_day")
            ).otherwise(f.col(f"customer_current_mileage"))
        ).withColumn(
            f"{cat}_is_due",
            f.when(
                (f.col(f"{cat}_days_until_change") - f.col(f"{cat}_days_since_last_trx")) < 0,
                1
            ).otherwise(0)
        ).withColumn(
            f"{cat}_mileage_when_due",
            f.when(
                f.col("customer_current_mileage").isNull() &
                (f.col(f"{cat}_is_due") == 1),
                f.col(f"customer_last_current_mileage") + f.col(f"{cat}_days_since_last_trx") * f.col(f"{cat}_avg_mileage_per_day")
            ).otherwise(f.col(f"customer_current_mileage"))
        ).withColumn(
            f"{cat}_is_due_40k",
            f.when(
                (((f.col(f"{cat}_mileage_when_due") / f.lit(40000)) - f.floor((f.col(f"{cat}_mileage_when_due") / f.lit(40000)))) > 0.94) |
                (((f.col(f"{cat}_mileage_when_due") / f.lit(40000)) - f.floor((f.col(f"{cat}_mileage_when_due") / f.lit(40000)))) < 0.06),
                f.lit(1)
            ).otherwise(f.lit(0))
        ).withColumn(
            f"{cat}_is_due_80k",
            f.when(
                (((f.col(f"{cat}_mileage_when_due") / f.lit(80000)) - f.floor((f.col(f"{cat}_mileage_when_due") / f.lit(80000)))) > 0.97) &
                (((f.col(f"{cat}_mileage_when_due") / f.lit(80000)) - f.floor((f.col(f"{cat}_mileage_when_due") / f.lit(80000)))) < 0.03),
                f.lit(1)
            ).otherwise(f.lit(0))
        ).withColumn(
            f"{cat}_is_due_branch",
            f.when(
                (f.col(f"{cat}_is_due") > 0) &
                ((f.col(f"{cat}_is_due_40k") > 0) |
                (f.col(f"{cat}_is_due_80k") > 0)),
                f.lit("PAC")
            ).when(
                (f.col(f"{cat}_is_due") > 0) &
                (f.col(f"{cat}_is_due_40k") < 1) &
                (f.col(f"{cat}_is_due_80k") < 1),
                f.lit("PE")
            ).otherwise(None)
        )

    w11 = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt")).rowsBetween(-11, 0)
    w_id_observ = Window.partitionBy("_id").orderBy("_observ_end_dt")

    ftr_sales = ftr_sales.withColumn(
        "month_avg_mileage_per_day_nullif_past_11_next_0_months",
        f.avg(
            f.replace(f.col("customer_avg_mileage_per_day"), f.lit(0.0))
        ).over(w11)
    ).withColumn(
        "estimated_days_to_change_mineral_oil",
        f.when(
            f.col("customer_current_mileage").isNotNull(),
            f.round(f.lit(5000) / f.col("month_avg_mileage_per_day_nullif_past_11_next_0_months")) # mineral oil
        ).otherwise(None)
    ).withColumn(
        "estimated_days_to_change_synthetic_oil",
        f.when(
            f.col("customer_current_mileage").isNotNull(),
            f.round(f.lit(10000) / f.col("month_avg_mileage_per_day_nullif_past_11_next_0_months")) # synthetic oil
        ).otherwise(None)
    ).withColumn(
        "estimated_days_to_change_mineral_oil",
        f.last("estimated_days_to_change_mineral_oil", ignorenulls=True).over(w_id_observ.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "estimated_days_to_change_synthetic_oil",
        f.last("estimated_days_to_change_synthetic_oil", ignorenulls=True).over(w_id_observ.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
    #     "last_transaction_dt",
    #     f.last("last_transaction_dt", ignorenulls=True).over(win_id_monthly.rangeBetween(Window.unboundedPreceding, 0))
    # ).withColumn(
        "days_since_oil_last_transactions",
        f.date_diff("_observ_end_dt", "oil_last_transaction_dt")
    ).withColumn(
        "days_since_oil_synthetic_last_transactions",
        f.date_diff("_observ_end_dt", "oil_synthetic_last_transaction_dt")
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

    # breakpoint()

    # ftr_sales.filter(f.col("_id") == "5205SLJ__966545078777").select("_id", "_observ_end_dt", "customer_current_mileage", "customer_last_current_mileage", "customer_avg_mileage_per_day", "month_avg_mileage_per_day_nullif_past_11_next_0_months", "days_since_oil_synthetic_last_transactions", "estimated_days_to_change_synthetic_oil", "days_until_synthetic_oil_change").orderBy("_id", "_observ_end_dt").show(60, truncate=False)

    w_churn = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt")).rowsBetween(1, 2)
    w_churn_acc = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt")).rowsBetween(-2, 0)

    win_id_monthly = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt"))

    out = ftr_sales.withColumn(
        "last_transaction_dt",
        f.last("last_transaction_dt", ignorenulls=True).over(win_id_monthly.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "customer_mineral_oil",
        f.when(
            f.col("month_net_sales_category_oil") > 0,
            1
        ).otherwise(None)
    ).withColumn(
        "customer_synthetic_oil",
        f.when(
            f.col("month_net_sales_category_oil_synthetic") > 0,
            1
        ).otherwise(None)
    ).withColumn(
        "customer_mineral_oil",
        f.when(
            f.col("customer_synthetic_oil") > 0,
            0
        ).otherwise(f.col("customer_mineral_oil"))
    ).withColumn(
        "customer_synthetic_oil",
        f.when(
            f.col("customer_mineral_oil") > 0,
            0
        ).otherwise(f.col("customer_synthetic_oil"))
    ).withColumn(
        "customer_mineral_oil",
        f.last("customer_mineral_oil", ignorenulls=True).over(win_id_monthly.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "customer_synthetic_oil",
        f.last("customer_synthetic_oil", ignorenulls=True).over(win_id_monthly.rangeBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "customer_mineral_oil",
        f.when(
            f.col("customer_mineral_oil").isNull(),
            0
        ).otherwise(f.col("customer_mineral_oil"))
    ).withColumn(
        "customer_synthetic_oil",
        f.when(
            f.col("customer_synthetic_oil").isNull(),
            0
        ).otherwise(f.col("customer_synthetic_oil"))
    ).withColumn(
        "is_churn",
        f.when(
            (f.max("is_active").over(win_id_monthly.rowsBetween(0, 1)) > 0),
            0
        ).when(
            (f.col("customer_mineral_oil") > 0) &
            (f.col("below_estimated_mineral_oil_change") > 0),
            0
        ).when(
            (f.col("customer_synthetic_oil") > 0) &
            (f.col("below_estimated_synthetic_oil_change") > 0),
            0
        ).otherwise(1)
    ).withColumn(
        "target_churn_1",
        f.max("is_churn").over(win_id_monthly.rowsBetween(1, 1))
    ).withColumn(
        "target_churn_2",
        f.max("is_churn").over(win_id_monthly.rowsBetween(2, 2))
    ).withColumn(
        "target_churn_3",
        f.max("is_churn").over(win_id_monthly.rowsBetween(3, 3))
    ).withColumn(
        "acc_churn_past_2_next_0_months",
        f.sum("is_churn").over(w_churn_acc)
    )

    return out.orderBy("_id", "_observ_end_dt")