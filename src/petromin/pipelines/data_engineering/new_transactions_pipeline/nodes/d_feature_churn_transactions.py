"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import typing as tp

import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window

# TODO: dissociate from ftr sales and adjust dependency only to base sales

def create_churn_features(
    spine: pyspark.sql.DataFrame,
    ftr_sales: pyspark.sql.DataFrame,
    ftr_mileage: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    w_churn = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt")).rowsBetween(1, 2)
    w_churn_acc = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt")).rowsBetween(-2, 0)

    win_id_monthly = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt"))

    base_churn = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_sales.select(
            "_id", "_observ_end_dt",
            "month_net_sales_category_oil",
            "month_net_sales_category_oil_synthetic",
            "is_active",
            "last_transaction_dt",
        ),
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_mileage.select(
            "_id", "_observ_end_dt",
            "is_due_mineral_oil",
            "is_due_synthetic_oil",
        ),
        on=["_id", "_observ_end_dt"],
        how="left"
    )


    out = base_churn.withColumn(
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
            (f.col("is_due_mineral_oil") < 1),
            0
        ).when(
            (f.col("customer_synthetic_oil") > 0) &
            (f.col("is_due_synthetic_oil") < 1),
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
    ).select(
        "_id",
        "_observ_end_dt",
        "is_churn",
        "target_churn_1",
        "target_churn_2",
        "target_churn_3",
        "acc_churn_past_2_next_0_months",
    )

    return out.orderBy(["_id", "_observ_end_dt"])
