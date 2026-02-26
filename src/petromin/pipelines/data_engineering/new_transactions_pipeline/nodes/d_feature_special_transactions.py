import typing as tp

import pyspark.sql.dataframe
import pandas as pd
from pyspark.sql import functions as f, Window



def create_special_trx_features(
    base_sales: pyspark.sql.DataFrame,
    prm_promo: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    promo_list = prm_promo.select("Promo").distinct().rdd.map(lambda r: r[0]).collect()

    ftr_pms_monthly_trx = base_sales.filter(
        f.col("is_pms") == f.lit(1)
    ).groupBy(
        ["_id", "_observ_end_dt"]
    ).agg(
        f.sum("sales_amount_net").alias("pms_month_net_sales"),
        f.countDistinct("transaction_id").alias("pms_month_distinct_transactions"),
    )

    ftr_promo_monthly_trx = base_sales.withColumn(
        "has_promo",
        f.when(
            f.upper("package_name").isin(*[promo.upper() for promo in promo_list]),
            f.lit(1)
        ).otherwise(f.lit(0))
    ).groupBy(
        ["_id", "_observ_end_dt", "transaction_id"]
    ).agg(
        f.sum("sales_amount_net").alias("sales_amount_net"),
        f.max("has_promo").alias("has_promo"),
    ).filter(
        f.col("has_promo") > f.lit(0)
    ).groupBy(
        ["_id", "_observ_end_dt",]
    ).agg(
        f.sum("sales_amount_net").alias("promo_month_net_sales"),
        f.countDistinct("transaction_id").alias("promo_month_distinct_transactions"),
    )

    ftr_promos_monthly_trx = base_sales.withColumn(
        "has_promo",
        f.when(
            f.upper("package_name").isin(*[promo.upper() for promo in promo_list]),
            f.lit(1)
        ).otherwise(f.lit(0))
    ).filter(
        f.col("has_promo") > f.lit(0)
    ).groupBy(
        ["_id", "_observ_end_dt", "transaction_id"]
    ).agg(
        f.concat_ws("| ", f.collect_list("package_name")).alias("all_packages")
    ).groupBy(
        ["_id", "_observ_end_dt",]
    ).agg(
        f.concat_ws(" | ", f.collect_list("all_packages")).alias("last_promo")
    )


    out = base_sales.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_pms_monthly_trx,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_promo_monthly_trx,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_promos_monthly_trx,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).fillna(0).select(
        "_id",
        "_observ_end_dt",
        "pms_month_net_sales",
        "pms_month_distinct_transactions",
        "promo_month_net_sales",
        "promo_month_distinct_transactions",
        "last_promo",
    )


    return out.orderBy(["_id", "_observ_end_dt"])
