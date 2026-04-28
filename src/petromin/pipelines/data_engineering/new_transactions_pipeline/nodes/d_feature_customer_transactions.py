"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import typing as tp

import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window


# TODO: dissociate from ftr sales and adjust dependency only to base sales

def create_segment_features(
    spine: pyspark.sql.DataFrame,
    ftr_sales: pyspark.sql.DataFrame,
    ftr_mileage: pyspark.sql.DataFrame,
    ftr_special_trxs: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    base_segment = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_sales.select(
            "_id", "_observ_end_dt",
            "month_net_sales",
            "month_net_sales_category_oil",
            "month_net_sales_category_oil_synthetic",
            "last_transaction_dt",
            "month_distinct_transactions",
            "month_total_qty_discounts",
            "smart_target_mineral",      # <-- NEW: Pulling the dynamic target
            "smart_target_synthetic",
        ),
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_mileage.select(
            "_id", "_observ_end_dt",
            "customer_mineral_oil",
            "customer_synthetic_oil",
            "customer_last_transaction_dt",
            "customer_avg_mileage_per_day",
        ),
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_special_trxs.select(
            "_id",
            "_observ_end_dt",
            "pms_month_net_sales",
            "pms_month_distinct_transactions",
            "promo_month_net_sales",
            "promo_month_distinct_transactions",
            "last_promo",
        ),
        on=["_id", "_observ_end_dt"],
        how="left"
    )

    win_id_transaction = Window.partitionBy("_id").orderBy(f.col("_observ_end_dt"))

    out = base_segment.withColumn(
        "first_visit_dt",
        f.first("customer_last_transaction_dt", ignorenulls=True).over(win_id_transaction.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "months_since_first_visit",
        f.floor(f.months_between(f.col("_observ_end_dt"), f.col("first_visit_dt")))
    ).withColumn(
        "months_since_last_visit",
        f.floor(f.months_between(f.col("_observ_end_dt"), f.col("customer_last_transaction_dt")))
    ).withColumn(
        "total_number_of_visits",
        f.sum("month_distinct_transactions").over(win_id_transaction.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "total_number_of_promo_visits",
        f.sum("promo_month_distinct_transactions").over(win_id_transaction.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "total_number_of_pms_visits",
        f.sum("pms_month_distinct_transactions").over(win_id_transaction.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "total_number_of_visits_last_12_months",
        f.sum("month_distinct_transactions").over(win_id_transaction.rowsBetween(-11, 0))
    ).withColumn(
        "total_number_of_promo_visits_last_12_months",
        f.sum("promo_month_distinct_transactions").over(win_id_transaction.rowsBetween(-11, 0))
    ).withColumn(
        "total_number_of_pms_visits_last_12_months",
        f.sum("pms_month_distinct_transactions").over(win_id_transaction.rowsBetween(-11, 0))
    ).withColumn(
        "ratio_promo_total_visits_last_12_months",
        f.round(f.col("total_number_of_promo_visits_last_12_months") / f.col("total_number_of_visits_last_12_months"), 2)
    ).withColumn(
        "ratio_pms_total_visits_last_12_months",
        f.round(f.col("total_number_of_pms_visits_last_12_months") / f.col("total_number_of_visits_last_12_months"), 2)
    ).withColumn(
        "total_revenue",
        f.sum("month_net_sales").over(win_id_transaction.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "total_promo_revenue",
        f.sum("promo_month_net_sales").over(win_id_transaction.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "total_pms_revenue",
        f.sum("pms_month_net_sales").over(win_id_transaction.rowsBetween(Window.unboundedPreceding, 0))
    ).withColumn(
        "total_revenue_last_12_months",
        f.sum("month_net_sales").over(win_id_transaction.rowsBetween(-11, 0))
    ).withColumn(
        "total_promo_revenue_last_12_months",
        f.sum("promo_month_net_sales").over(win_id_transaction.rowsBetween(-11, 0))
    ).withColumn(
        "total_pms_revenue_last_12_months",
        f.sum("pms_month_net_sales").over(win_id_transaction.rowsBetween(-11, 0))
    ).withColumn(
        "expected_number_of_visits_mineral_oil",
        # Replaced 5000 with the dynamic customer-specific target
        f.floor((f.col("customer_avg_mileage_per_day") * f.lit(365)) / f.col("smart_target_mineral"))
    ).withColumn(
        "expected_number_of_visits_synthetic_oil",
        # Replaced 10000 with the dynamic customer-specific target
        f.floor((f.col("customer_avg_mileage_per_day") * f.lit(365)) / f.col("smart_target_synthetic"))
    ).withColumn(
        "is_new_joiner",
        f.when(
            (f.col("total_number_of_visits") == 1) &
            (f.col("months_since_last_visit") < 12) &
            (f.col("months_since_first_visit") < 12),
            1
        ).otherwise(0)
    ).withColumn(
        "is_loyal",
        f.when(
            (f.col("total_number_of_visits_last_12_months") >= 3) &
            (((f.col("customer_mineral_oil") > 0) & (f.col("total_number_of_visits_last_12_months") >= f.col("expected_number_of_visits_mineral_oil"))) |
            ((f.col("customer_synthetic_oil") > 0) & (f.col("total_number_of_visits_last_12_months") >= f.col("expected_number_of_visits_synthetic_oil")))),
            1
        ).otherwise(0)
    ).withColumn(
        "is_potential_loyal",
        f.when(
            (f.col("total_number_of_visits_last_12_months") >= 2) &
            (((f.col("customer_mineral_oil") > 0) & (f.col("total_number_of_visits_last_12_months") > (f.col("expected_number_of_visits_mineral_oil") * 0.7))) |
            ((f.col("customer_synthetic_oil") > 0) & (f.col("total_number_of_visits_last_12_months") > (f.col("expected_number_of_visits_synthetic_oil") * 0.7)))) &
            (f.col("is_loyal") == 0),
            1
        ).otherwise(0)
    ).withColumn(
        "is_uncommited",
        f.when(
            (f.col("total_number_of_visits_last_12_months") >= 1) &
            (f.col("months_since_last_visit") < 12) &
            (f.col("is_new_joiner") == 0) &
            (f.col("is_loyal") == 0) &
            (f.col("is_potential_loyal") == 0),
            1
        ).otherwise(0)
    ).withColumn(
        "is_lost",
        f.when(
            (f.col("months_since_last_visit") > 11) &
            (f.col("months_since_last_visit") <= 24),
            1
        ).otherwise(0)
    ).withColumn(
        "is_gone",
        f.when(
            (f.col("months_since_last_visit") > 24),
            1
        ).otherwise(0)
    ).withColumn(
        "is_promo_hunter",
        f.when(
            (f.col("total_number_of_visits_last_12_months") > 0) &
            (f.col("total_number_of_visits_last_12_months") == f.col("total_number_of_promo_visits_last_12_months")),
            1
        ).otherwise(0)
    ).withColumn(
        "is_full_price",
        f.when(
            (f.col("total_number_of_visits_last_12_months") > 0) &
            (f.col("total_number_of_promo_visits_last_12_months") == 0),
            1
        ).otherwise(0)
    ).withColumn(
        "is_mixed_price",
        f.when(
            (f.col("is_promo_hunter") < 1) &
            (f.col("is_full_price") < 1),
            1
        ).otherwise(0)
    ).select(
        "_id",
        "_observ_end_dt",
        # "customer_mineral_oil",
        # "customer_synthetic_oil",
        "first_visit_dt",
        "months_since_first_visit",
        "months_since_last_visit",
        "total_number_of_visits",
        "total_number_of_promo_visits",
        "total_number_of_pms_visits",
        "total_number_of_visits_last_12_months",
        "total_number_of_promo_visits_last_12_months",
        "total_number_of_pms_visits_last_12_months",
        "ratio_promo_total_visits_last_12_months",
        "ratio_pms_total_visits_last_12_months",
        "total_revenue",
        "total_promo_revenue",
        "total_pms_revenue",
        "total_revenue_last_12_months",
        "total_promo_revenue_last_12_months",
        "total_pms_revenue_last_12_months",
        "expected_number_of_visits_mineral_oil",
        "expected_number_of_visits_synthetic_oil",
        "is_new_joiner",
        "is_loyal",
        "is_potential_loyal",
        "is_uncommited",
        "is_lost",
        "is_gone",
        "is_promo_hunter",
        "is_full_price",
        "is_mixed_price",
    )

    return out.orderBy(["_id", "_observ_end_dt"])
