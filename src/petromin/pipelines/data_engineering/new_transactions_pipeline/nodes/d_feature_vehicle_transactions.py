"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

import typing as tp

import pyspark.sql.dataframe
from pyspark.sql import functions as f, Window
from pyspark.sql.types import StringType

from .c_primary_spine import create_auxillary_columns


def create_customer_vehicle_features(
    spine: pyspark.sql.DataFrame,
    base_sales: pyspark.sql.DataFrame,
    prm_customers: pyspark.sql.DataFrame,
    prm_vehicles: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:

    def categorizer(age):
        if age is None:
            return None
        elif age < 3:
            return "0_2"
        elif age < 6:
            return "3_5"
        elif age < 10:
            return "6_9"
        else:
            return "10_plus"

    bucket_udf = f.udf(categorizer, StringType() )

    prm_customer_vehicles = prm_customers.join(
        prm_vehicles,
        on="customer_id",
        how="inner"
    )

    prm_customer_vehicles = create_auxillary_columns(
        prm_customer_vehicles, unit_of_analysis=["plate_number", "mobile"]
    )

    prm_customer_vehicles = prm_customer_vehicles.groupBy(
        "_id"
    ).agg(
        f.mode("nationality").alias("nationality"),
        f.mode("preferred_language").alias("preferred_language"),
        f.mode("maker").alias("maker"),
        f.mode("model").alias("model"),
        f.mode("model_year").alias("model_year"),
        f.mode("vehicle_brand_level").alias("vehicle_brand_level"),
        f.mode("is_truck").alias("is_truck"),
    )

    ftr_vehicles = base_sales.select(
        "_id", "_observ_end_dt", "transaction_id", "transaction_dt"
    ).distinct().join(
        prm_customer_vehicles,
        on=["_id"],
        how="left"
    )

    ftr_vehicles = ftr_vehicles.withColumn(
        "car_age",
        f.when(
            (f.col("model_year") >= 1980),
            f.year(f.col("_observ_end_dt")).cast("int") - f.col("model_year").cast("int")
        ).otherwise(f.lit(-1))
    ).withColumn(
        "car_age_group",
        bucket_udf(f.col("car_age"))
    ).withColumn(
        "car_group",
        f.concat(f.lower(f.col("vehicle_brand_level")), f.lit("__"), f.col("car_age_group"))
    )

    car_price_levels = [f"car_brand__{col}" for col in ["very_low", "low", "medium", "high", "very_high"]]
    car_age_levels = [f"car_age__{col}" for col in [None, "0_2", "3_5", "6_9", "10_plus"]]
    car_price_age_levels = [
        f"{price}__{age}"
        for price in car_price_levels
        for age in car_age_levels
    ]

    ftr_vehicles_price_pivot = ftr_vehicles.dropna(
        subset=["_id", "transaction_id"]
    ).withColumn(
        "vehicle_brand_level",
        f.concat(f.lit("car_brand__"), f.lower(f.col("vehicle_brand_level")))
    ).groupBy(
        "_id", "transaction_id"
    ).pivot(
        "vehicle_brand_level", values=car_price_levels
    ).agg(
        f.countDistinct("_id").alias("count")
    ).fillna(0)

    ftr_vehicles_age_pivot = ftr_vehicles.dropna(
        subset=["_id", "transaction_id"]
    ).withColumn(
        "car_age_group",
        f.concat(f.lit("car_age__"), f.col("car_age_group"))
    ).groupBy(
        "_id", "transaction_id"
    ).pivot(
        "car_age_group", values=car_age_levels
    ).agg(
        f.countDistinct("_id").alias("count")
    ).fillna(0)

    ftr_vehicles_extended = ftr_vehicles.join(
        ftr_vehicles_price_pivot,
        on=["_id", "transaction_id"],
        how="left"
    ).join(
        ftr_vehicles_age_pivot,
        on=["_id", "transaction_id"],
        how="left"
    )

    ftr_vehicles_selected = ftr_vehicles_extended.withColumn(
        "is_toyota",
        f.when(f.lower("maker") == "toyota", 1).otherwise(0)
    ).withColumn(
        "is_japanese",
        f.when(f.lower("maker").isin(["toyota", "nissan", "hyundai", "lexus", "honda", "mazda", "suzuki", "daihatsu"]), 1).otherwise(0)
    ).withColumn(
        "is_american",
        f.when(f.lower("maker").isin(["ford", "chevrolet", "mercury"]), 1).otherwise(0)
    ).withColumn(
        "is_chinese",
        f.when(
            f.lower("maker").isin(
                ["gmc", "byd", "geely", "chery", "gwm", "faw", "jac", "mg",
                 "gac", "jetour", "haval", "changan", "isuzu"]
            ),
            1
        ).otherwise(0)
    ).withColumn(
        "is_chinese_brand",
        f.when(
            f.upper("maker").isin(['CHANGAN', 'GEELY', 'HAVAL', 'MG']),
            f.lit(1)
        ).otherwise(f.lit(0))
    ).select(
        "_id",
        "_observ_end_dt",
        "nationality",
        "preferred_language",
        "maker",
        "model",
        "model_year",
        "car_age",
        "car_age_group",
        *car_price_levels,
        *car_age_levels,
        "is_toyota",
        "is_japanese",
        "is_american",
        "is_chinese",
        "is_chinese_brand",
        "is_truck"
    ).orderBy("_id", "_observ_end_dt",)

    out = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        ftr_vehicles_selected.drop_duplicates(subset=["_id", "_observ_end_dt"]),
        on=["_id", "_observ_end_dt"],
        how="left"
    )

    return out

