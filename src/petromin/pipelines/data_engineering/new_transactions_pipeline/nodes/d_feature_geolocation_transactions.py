"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import logging
import typing as tp

import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window

logger = logging.getLogger(__name__)


def create_geolocation_features(
    spine: pyspark.sql.DataFrame,
    base_sales: pyspark.sql.DataFrame,
    prm_geolocation: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    sales_branch_view = base_sales.groupBy(
        ["_id", "_observ_end_dt", "branch_id"]
    ).agg(
        f.countDistinct("transaction_id").alias("trx_per_branch")
    )

    sales_branch_geolocation = sales_branch_view.join(
        prm_geolocation,
        on="branch_id",
        how="left"
    )

    w = Window.partitionBy('_id', '_observ_end_dt').orderBy(f.col('trx_per_branch').desc())

    geolocations_cols_to_drop = ["trx_per_branch", "longitude", "latitude","branch_code","branch_type","is_active","city"]
    geolocations_cols_to_keep = [
        col
        for col in sales_branch_geolocation.columns
        if col not in geolocations_cols_to_drop
    ]

    select_list = [
        f.col(col).alias(f"most_trx_branch_{col}")
        for col in geolocations_cols_to_keep
        if col not in ["_id", "_observ_end_dt",]
    ]

    ftr_branch_most_trasactions_geolocation = sales_branch_geolocation.withColumn(
        'row_number',
        f.row_number().over(w)
    ).filter(
        f.col("trx_per_branch") > 0
    ).orderBy(
        "_id", "_observ_end_dt", "row_number"
    ).filter(
        f.col('row_number') == 1
    ).select("_id", "_observ_end_dt", *select_list)

    agg_list = [
        f.avg(f.col(col)).alias(f"avg_{col}")
        for col in geolocations_cols_to_keep
        if col not in ["_id", "_observ_end_dt", "branch_id"]
    ]

    ftr_avg_branch_geolocation = sales_branch_geolocation.filter(
        f.col("trx_per_branch") > 0
    ).drop(
        *geolocations_cols_to_drop
    ).groupby(
        "_id", "_observ_end_dt"
    ).agg(*agg_list)

    out = spine.select("_id", "_observ_end_dt").distinct().join(
        ftr_branch_most_trasactions_geolocation,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        ftr_avg_branch_geolocation,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).orderBy("_id", "_observ_end_dt")

    logger.info(
        f"geolocation dataframe:\t\t\t({out.count()}, {len(out.columns)})",
    )

    return out
