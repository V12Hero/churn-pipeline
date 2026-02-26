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


def create_branches_features(
    spine: pyspark.sql.DataFrame,
    base_sales: pyspark.sql.DataFrame,
    prm_branches: pyspark.sql.DataFrame,
    ftr_mileage: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    ftr_trx_branch = base_sales.groupBy(
        ["transaction_id",]
    ).agg(
        f.mode("branch_id").alias("branch_id")
    ).withColumnRenamed(
        "transaction_id", "customer_last_transaction_id"
    ).orderBy(f.desc("customer_last_transaction_id"))

    ftr_trx_city = ftr_trx_branch.join(
        prm_branches.select("branch_id", "branch_code", "city").distinct(),
        on="branch_id",
        how="left"
    ).select("customer_last_transaction_id", "city", "branch_code")

    ftr_branches = ftr_mileage.join(
        ftr_trx_city,
        on="customer_last_transaction_id",
        how="left"
    )

    ftr_branches = ftr_branches.withColumn(
        "is_pac_city",
        f.when(
            f.upper("city").isin(
                "ABHA",
                "BURAIDAH",
                "DAMMAM",
                "DWADMI",
                "DUBA",
                "HASSA",
                "JEDDAH",
                "JIZAN",
                "JUBAIL",
                "KHAFJI",
                "KHAMIS",
                "KHARJ",
                "KHOBAR",
                "MADINAH",
                "MAKKAH",
                "MUHYAIL",
                "RIYADH",
                "SAKAKA",
                "TABUK",
                "TAIF",
                "UNAIZAH",
                "YANBU"
            ),
            1
        ).otherwise(0)
    )

    out = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_branches,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).select(
        "_id", "_observ_end_dt",
        # "customer_last_transaction_id",
        "city",
        "branch_code",
        "is_pac_city",
    )

    return out.orderBy(["_id", "_observ_end_dt"])
