"""
This is a boilerplate pipeline 'build_feature_layer'
generated using Kedro 0.18.8
"""

from typing import List
import numpy as np
import pandas as pd
import gc
import pyspark.sql.dataframe
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as f, Window


import logging

logger = logging.getLogger(__name__)


def filter_with_conditions(
        df:pyspark.sql.DataFrame,
        conditions: List[str]
)->pyspark.sql.DataFrame:
    filter_cond= f.lit(True)

    # breakpoint()

    if conditions:
        for cond in conditions:
            msg = f"adding condition {cond}"
            logger.info(msg)
            filter_cond = filter_cond & (f.expr(cond))

    out = df.filter(filter_cond)

    logger.info(
        f"filtered master shape:  ({out.count()}, {len(out.columns)})",
    )

    # breakpoint()


    return out.orderBy(["_id","_observ_end_dt"])


def prepare_mileage_forecast(
    ftr_master: pyspark.sql.DataFrame,
) -> pyspark.sql.DataFrame:
    
    client_mileage_product_forecast = ftr_master.withColumn(
        "mobile",
        f.split("_id", "__")[1]
    ).withColumn(
        "reference_digits",
        f.substring("mobile", -2, 2)
    ).withColumn(
        "campaign_group",
        f.when(
            f.col("reference_digits").isin("00", "01", "02", "03", "04"),
            "control"
        ).otherwise("test")
    )

    # breakpoint()

    [col for col in client_mileage_product_forecast.columns if "customer" in col]

    ftr_mobile_trx = client_mileage_product_forecast.groupBy(
        "mobile", "_observ_end_dt"
    ).agg(
        f.sum("total_number_of_visits").alias("mobile_total_number_of_visits"),
        f.sum("total_number_of_promo_visits").alias("mobile_total_number_of_promo_visits"),
        f.sum("total_number_of_pms_visits").alias("mobile_total_number_of_pms_visits"),
        f.sum("total_number_of_visits_last_12_months").alias("mobile_total_number_of_visits_last_12_months"),
        f.sum("total_number_of_promo_visits_last_12_months").alias("mobile_total_number_of_promo_visits_last_12_months"),
        f.sum("total_number_of_pms_visits_last_12_months").alias("mobile_total_number_of_pms_visits_last_12_months"),
        f.sum("total_revenue").alias("mobile_total_revenue"),
        f.sum("total_promo_revenue").alias("mobile_total_promo_revenue"),
        f.sum("total_pms_revenue").alias("mobile_total_pms_revenue"),
        f.sum("total_revenue_last_12_months").alias("mobile_total_revenue_last_12_months"),
        f.sum("total_promo_revenue_last_12_months").alias("mobile_total_promo_revenue_last_12_months"),
        f.sum("total_pms_revenue_last_12_months").alias("mobile_total_pms_revenue_last_12_months"),
        f.max("customer_last_transaction_dt").alias("mobile_last_transaction_dt"),
        f.max("total_revenue").alias("mobile_highest_revenue"),
    )

    client_mileage_product_forecast_with_mobile = client_mileage_product_forecast.join(
        ftr_mobile_trx,
        on=["mobile", "_observ_end_dt"],
        how="left"
    )
    
    client_mileage_product_forecast_with_mobile = client_mileage_product_forecast_with_mobile.withColumn(
        "is_highest_revenue_car",
        f.when(
            f.col("total_revenue") == f.col("mobile_highest_revenue"),
            1
        ).otherwise(0)
    ).withColumn(
        "customer_mileage_forecast_bucket",
        f.floor(f.col("customer_mileage_forecast") / f.lit(10000)) * f.lit(10000)
    ).withColumn(
        "customer_mileage_last_forecast_bucket",
        f.floor(f.col("customer_mileage_last_forecast") / f.lit(10000)) * f.lit(10000)
    ).withColumn(
        "crossing_10k_threshold",
        f.when(
            f.col("customer_mileage_last_forecast_bucket") < f.col("customer_mileage_forecast_bucket"),
            1
        ).otherwise(0)
    ).withColumn(
        "is_due",
        f.lit("")
    )

    for product in ["Mineral Oil", "Synthetic Oil", "AC Services", "Battery", "Engine Flush", "Tires", "Air Filter", "Cabin AC Filter", "Coolant", "Spark Plugs"]:
        client_mileage_product_forecast_with_mobile = client_mileage_product_forecast_with_mobile.withColumn(
        "is_due",
        f.when(
            (f.col(f"is_due_{product.replace(" ", "_").lower()}") > 0),
            f.col("is_due") + f.lit(f"__{product.replace(" ", "_").lower()}")
        ).otherwise(f.col("is_due"))
    )

    # breakpoint()

    # [col for col in client_mileage_product_forecast_with_mobile.columns if "is_due" in col]

    # trx_promos = base_sales.withColumn(
    #     "has_promo",
    #     f.when(
    #         f.upper("package_name").isin(*[promo.upper() for promo in promo_list]),
    #         f.lit(1)
    #     ).otherwise(f.lit(0))
    # ).filter(
    #     f.col("has_promo") > f.lit(0)
    # ).groupBy(
    #     ["transaction_id", "package_name"]
    # ).agg(
    #     f.countDistinct("product_id").alias("count")
    # ).groupBy(
    #     ["transaction_id"]
    # ).agg(
    #     f.concat_ws(" | ", f.collect_list("package_name")).alias("last_promo")
    # ).withColumnRenamed(
    #     "transaction_id", "customer_last_transaction_id"
    # ).orderBy(f.desc("customer_last_transaction_id"))

    # client_mileage_product_forecast = client_mileage_product_forecast.join(
    #     trx_promos,
    #     on=["customer_last_transaction_id"],
    #     how="left"
    # )

    out = client_mileage_product_forecast_with_mobile

    return out.orderBy("_id", "_observ_end_dt")


# def forecast_mileage(
#         forecast_df: pd.DataFrame,
#         churn_df: pd.DataFrame,
#         # oem_rules_df: pd.DataFrame,
#         closed_station_list: List,
# ) -> pd.DataFrame:

    
#     logger.info("Creating Features - Expected visits ")
#     forecast_df["customer_expected_visits"] = np.where(
#         forecast_df["customer_mineral_oil"] == 1,
#         forecast_df["expected_number_of_visits_mineral_oil"],
#         forecast_df["expected_number_of_visits_synthetic_oil"]
#     )

#     logger.info("Creating Features - Loyalty segments")
#     forecast_df["loyalty_segment"] = None
#     forecast_df["loyalty_segment"] = np.where(forecast_df["is_new_joiner"]==1, "New Joiner",forecast_df["loyalty_segment"])
#     forecast_df["loyalty_segment"] = np.where(forecast_df["is_uncommited"]==1, "Uncommited",forecast_df["loyalty_segment"])
#     forecast_df["loyalty_segment"] = np.where(forecast_df["is_potential_loyal"]==1, "Potential Loyal",forecast_df["loyalty_segment"])
#     forecast_df["loyalty_segment"] = np.where(forecast_df["is_loyal"]==1, "Loyal",forecast_df["loyalty_segment"])
#     forecast_df["loyalty_segment"] = np.where(forecast_df["is_lost"]==1, "Lost",forecast_df["loyalty_segment"])
#     forecast_df["loyalty_segment"] = np.where(forecast_df["is_gone"]==1, "Gone",forecast_df["loyalty_segment"])

#     logger.info("Creating Features - Price segments")
#     forecast_df["price_segment"] = None
#     forecast_df["price_segment"] = np.where(forecast_df["is_full_price"]==1, "Full price",forecast_df["price_segment"])
#     forecast_df["price_segment"] = np.where(forecast_df["is_promo_hunter"]==1, "Promo hunter",forecast_df["price_segment"])
#     forecast_df["price_segment"] = np.where(forecast_df["is_mixed_price"]==1, "Mixed price",forecast_df["price_segment"])

#     logger.info("Creating Features - Buckets")
#     forecast_df["customer_expected_visits_bucket"] = pd.cut(
#         forecast_df["customer_expected_visits"],
#         bins=[np.NINF, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.PINF],
#         labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10+"]
#     )

#     forecast_df["total_number_of_visits_last_12_months_bucket"] = pd.cut(
#         forecast_df["total_number_of_visits_last_12_months"],
#         bins=[np.NINF, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.PINF],
#         labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10+"]
#     )

#     forecast_df["decile"] = -1
#     forecast_df.loc[forecast_df["total_revenue_last_12_months"] > 0, "decile"] = forecast_df[forecast_df["total_revenue_last_12_months"] > 0].groupby(['_observ_end_dt'])['total_revenue_last_12_months'].transform(lambda x: pd.qcut(x, 10, labels=np.arange(10, 0, -1)))

#     forecast_df["ratio_pms_total_visits_last_12_months"] = forecast_df["total_number_of_pms_visits_last_12_months"] / forecast_df["total_number_of_visits_last_12_months"]

#     forecast_df["ratio_pms_total_visits_last_12_months_bucket"] = pd.cut(
#         forecast_df["ratio_pms_total_visits_last_12_months"],
#         bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1],
#         labels=["0.0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
#         right=True
#     )

#     forecast_df["car_age_bucket"] = pd.cut(
#         forecast_df["car_age"],
#         bins=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.PINF],
#         labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10+"],
#         right=False
#     )

#     forecast_df["mpd_bucket"] = pd.cut(
#         forecast_df["customer_mileage_per_day"],
#         bins=[0, 25, 50, 75, 100, 125, 150, 175, 200, 300, 500, 1000, np.PINF],
#         labels=["0-25", "25-50", "50-75", "75-100", "100-125", "125-150", "150-175", "175-200", "200-300", "300-500", "500-1000", "1000+"]
#     )

#     forecast_df["is_station_closed"] = np.where(
#         forecast_df["branch_code"].isin([str(station) for station in closed_station_list]),
#         1,
#         0
#     )

#     logger.info("Adjust Churn dataframes dates")

#     churn_df["_observ_end_dt"] = (churn_df["_observ_end_dt"] + pd.DateOffset(days=1) + pd.offsets.MonthEnd())

#     churn_df["_observ_end_dt"] = churn_df["_observ_end_dt"].astype("str")
#     forecast_df["_observ_end_dt"] = forecast_df["_observ_end_dt"].astype("str")

#     # breakpoint()
    
#     logger.info("Merge dataframes")
#     extended_forecast_df = pd.merge(
#         forecast_df,
#         churn_df[["_id", "_observ_end_dt", "churn_probability", "churn_bucket"]],
#         on=["_id", "_observ_end_dt"],
#         how="left"
#     )

#     logger.info("Delete original dataframes")
#     del forecast_df
#     del churn_df

#     logger.info("Create new features")

#     extended_forecast_df["is_toyota"] = np.where(extended_forecast_df["maker"].str.lower() == "toyota", 1, 0)

#     extended_forecast_df["is_due_mineral_synthetic"] = np.where(
#         (extended_forecast_df["is_due_mineral_oil"] == 1) | (extended_forecast_df["is_due_synthetic_oil"] == 1),
#         1, 
#         0
#     )

#     extended_forecast_df["is_on_warranty_period"] = np.where(
#         ((extended_forecast_df["maker"].str.lower().isin(["cadillac", "lexus"])) & ((extended_forecast_df["car_age"] >= 0) & (extended_forecast_df["car_age"] < 5))) |
#         ((extended_forecast_df["maker"].str.lower().isin(["hyundai", "kia", "changan", "renault", "jetour", "land rover"])) & ((extended_forecast_df["car_age"] >= 0) & (extended_forecast_df["car_age"] < 6))) |
#         ((extended_forecast_df["maker"].str.lower().isin(["mg", "chery", "haval"])) & ((extended_forecast_df["car_age"] >= 0) & (extended_forecast_df["car_age"] < 7))) |
#         ((extended_forecast_df["maker"].str.lower().isin(["geely"])) & ((extended_forecast_df["car_age"] >= 0) & (extended_forecast_df["car_age"] < 8))) |
#         ((~extended_forecast_df["maker"].str.lower().isin(["cadillac", "lexus", "hyundai", "kia", "changan", "renault", "jetour", "land rover", "mg", "chery", "haval", "geely", ])) & ((extended_forecast_df["car_age"] >= 0) & (extended_forecast_df["car_age"] < 4))),
#         1,
#         0
#     )

#     extended_forecast_df["is_pms_final_flag"] = np.where(
#         (extended_forecast_df["total_number_of_pms_visits_last_12_months"] > 0) &
#         (extended_forecast_df["is_on_warranty_period"] > 0) &
#         (extended_forecast_df["maker"].str.lower().isin(["geely", "haval", "hyundai", "kia", "mazda", "mg", "nissan", "toyota"])),
#         1,
#         0
#     )

#     logger.info("Filter final dataframe")
#     columns_to_keep = [
#         "is_pms_final_flag", "is_on_warranty_period",
#         "_id","mobile",'_observ_end_dt',
#         "PlateNumber",
#         "customer_last_transaction_dt",
#         "customer_days_since_last_trx",
#         "is_due_mineral_oil",
#         "is_due_synthetic_oil",
#         "nationality",
#         "maker",
#         "model",
#         "expected_number_of_visits_mineral_oil",
#         "expected_number_of_visits_synthetic_oil",
#         "is_due_PE",
#         "is_due_PAC",
#         "is_pac_city",
#         "is_due",
#         "preferred_language",
#         "is_chinese_brand",
#         "car_age",
#         "loyalty_segment",
#         "price_segment",
#         "campaign_group",
#         "decile",
#         "is_station_closed",
#         "churn_probability",
#         "churn_bucket",
#         'is_due_cabin_ac_filter', 'is_due_air_filter', 'is_due_fuel_additive',
#         'is_due_brake_fluid', 'is_due_coolant', 'is_due_differential_oil',
#         'is_due_fuel_filter', 'is_due_spark_plugs', 'is_due_atf',
#         'is_due_cvt_oil', 'is_due_drive_belt', 'is_due_canister',
#         'is_due_timing_belt', 'is_due_power_steering_pump_belt',
#         'is_due_atf_filter', 'is_due_pms', 'is_due_tires',
#         'is_due_engine_flush', 'is_due_battery', 'is_due_ac_services','is_due_mineral_oil',
#         'is_due_synthetic_oil','is_due_headlight_polishing', 'is_due_suspension','is_due_PE', 'is_due_PAC','is_due_mineral_synthetic',
#     ]
#     extended_forecast_df = extended_forecast_df[columns_to_keep]
#     print("Reached Extended Forecast Filter - filtered columns")

#     last_month = extended_forecast_df["_observ_end_dt"].max()
#     filter_date_2025 = extended_forecast_df["_observ_end_dt"] == last_month
#     print("Created filter for last month based on max data")
#     filtered_forecast_df = extended_forecast_df[filter_date_2025]
#     del extended_forecast_df
#     print("done all - at out variable")

#     # out = filtered_forecast_df.sort_values(['_id', '_observ_end_dt'])
#     # del filtered_forecast_df
#     # breakpoint()

#     return [filtered_forecast_df]

def forecast_mileage(
        forecast_df: pd.DataFrame,
        churn_df: pd.DataFrame,
        closed_station_list: list,
) -> pd.DataFrame:

    # --- helpers: predeclare category dtypes to reduce per-row string duplication ---
    _visits_cats = pd.CategoricalDtype(
        categories=["0","1","2","3","4","5","6","7","8","9","10","10+"], ordered=True
    )
    _ratio_cats  = pd.CategoricalDtype(
        categories=["0.0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"], ordered=True
    )
    _mpd_cats    = pd.CategoricalDtype(
        categories=[
            "0-25","25-50","50-75","75-100","100-125","125-150",
            "150-175","175-200","200-300","300-500","500-1000","1000+"
        ],
        ordered=True
    )
    _age_cats    = pd.CategoricalDtype(
        categories=["0","1","2","3","4","5","6","7","8","9","10","10+"], ordered=True
    )

    # cast common flag columns smaller (no value change)
    def _downcast_flags(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int8")

    _flag_cols = [
        "customer_mineral_oil","is_new_joiner","is_uncommited","is_potential_loyal","is_loyal","is_lost","is_gone",
        "is_full_price","is_promo_hunter","is_mixed_price",
        "is_due_mineral_oil","is_due_synthetic_oil",
        "total_number_of_pms_visits_last_12_months"
    ]
    _downcast_flags(forecast_df, [c for c in _flag_cols if c in forecast_df.columns])

    logger.info("Creating Features - Expected visits ")
    mineral = (forecast_df["customer_mineral_oil"].to_numpy() == 1)
    visits = np.where(
        mineral,
        forecast_df["expected_number_of_visits_mineral_oil"].to_numpy(),
        forecast_df["expected_number_of_visits_synthetic_oil"].to_numpy()
    )
    forecast_df["customer_expected_visits"] = visits
    del mineral, visits
    gc.collect()

    logger.info("Creating Features - Loyalty segments")
    loy_conds = [
        forecast_df["is_new_joiner"].to_numpy() == 1,
        forecast_df["is_uncommited"].to_numpy() == 1,
        forecast_df["is_potential_loyal"].to_numpy() == 1,
        forecast_df["is_loyal"].to_numpy() == 1,
        forecast_df["is_lost"].to_numpy() == 1,
        forecast_df["is_gone"].to_numpy() == 1,
    ]
    loy_vals = ["New Joiner","Uncommited","Potential Loyal","Loyal","Lost","Gone"]
    forecast_df["loyalty_segment"] = pd.Categorical(np.select(loy_conds, loy_vals, default=None), categories=loy_vals)

    logger.info("Creating Features - Price segments")
    price_conds = [
        forecast_df["is_full_price"].to_numpy() == 1,
        forecast_df["is_promo_hunter"].to_numpy() == 1,
        forecast_df["is_mixed_price"].to_numpy() == 1,
    ]
    price_vals = ["Full price","Promo hunter","Mixed price"]
    forecast_df["price_segment"] = pd.Categorical(np.select(price_conds, price_vals, default=None), categories=price_vals)

    del loy_conds, loy_vals, price_conds, price_vals
    gc.collect()

    logger.info("Creating Features - Buckets")
    forecast_df["customer_expected_visits_bucket"] = (
        pd.cut(
            forecast_df["customer_expected_visits"],
            bins=[np.NINF,0,1,2,3,4,5,6,7,8,9,10,np.PINF],
            labels=_visits_cats.categories,
            right=True,
            include_lowest=True,
        ).astype(_visits_cats)
    )

    forecast_df["total_number_of_visits_last_12_months_bucket"] = (
        pd.cut(
            forecast_df["total_number_of_visits_last_12_months"],
            bins=[np.NINF,0,1,2,3,4,5,6,7,8,9,10,np.PINF],
            labels=_visits_cats.categories,
            right=True,
            include_lowest=True,
        ).astype(_visits_cats)
    )

    # decile (keep as pandas nullable Int8)
    forecast_df["decile"] = pd.Series([-1] * len(forecast_df), index=forecast_df.index, dtype="Int8")
    m_posrev = forecast_df["total_revenue_last_12_months"].to_numpy() > 0
    if m_posrev.any():
        grp = forecast_df.loc[m_posrev].groupby("_observ_end_dt", sort=False)["total_revenue_last_12_months"]
        dec = grp.transform(lambda x: pd.qcut(x, 10, labels=np.arange(10, 0, -1))).astype("Int8")
        forecast_df.loc[m_posrev, "decile"] = dec
        del dec, grp
    del m_posrev
    gc.collect()

    # ratios/buckets
    with np.errstate(divide="ignore", invalid="ignore"):
        num = forecast_df["total_number_of_pms_visits_last_12_months"].to_numpy(dtype="float32")
        den = forecast_df["total_number_of_visits_last_12_months"].to_numpy(dtype="float32")
        ratio = num / np.where(den == 0, np.nan, den)
    forecast_df["ratio_pms_total_visits_last_12_months"] = ratio
    forecast_df["ratio_pms_total_visits_last_12_months_bucket"] = (
        pd.cut(
            forecast_df["ratio_pms_total_visits_last_12_months"],
            bins=[-1,0,0.2,0.4,0.6,0.8,1],
            labels=_ratio_cats.categories,
            right=True,
            include_lowest=True,
        ).astype(_ratio_cats)
    )
    del ratio
    gc.collect()

    forecast_df["car_age_bucket"] = (
        pd.cut(
            forecast_df["car_age"],
            bins=[-1,0,1,2,3,4,5,6,7,8,9,10,np.PINF],
            labels=_age_cats.categories,
            right=False,
            include_lowest=True,
        ).astype(_age_cats)
    )

    forecast_df["mpd_bucket"] = (
        pd.cut(
            forecast_df["customer_mileage_per_day"],
            bins=[0,25,50,75,100,125,150,175,200,300,500,1000,np.PINF],
            labels=_mpd_cats.categories,
            include_lowest=True,
        ).astype(_mpd_cats)
    )

    # closed stations
    logger.info("Creating Features - Station closed flag")
    closed_set = set(map(str, closed_station_list))
    forecast_df["is_station_closed"] = forecast_df["branch_code"].astype("string").isin(closed_set).astype("Int8")
    del closed_set
    gc.collect()

    logger.info("Adjust Churn dataframes dates")
    churn_df["_observ_end_dt"] = (churn_df["_observ_end_dt"] + pd.DateOffset(days=1) + pd.offsets.MonthEnd())
    churn_df["_observ_end_dt"] = churn_df["_observ_end_dt"].astype("str")
    forecast_df["_observ_end_dt"] = forecast_df["_observ_end_dt"].astype("str")

    logger.info("Merge dataframes")
    churn_cols = ["_id","_observ_end_dt","churn_probability","churn_bucket"]
    extended_forecast_df = forecast_df.merge(churn_df[churn_cols], on=["_id","_observ_end_dt"], how="left", copy=False)

    logger.info("Delete original dataframes")
    del forecast_df, churn_df
    gc.collect()

    logger.info("Create new features")
    maker_l = extended_forecast_df["maker"].astype("string").str.lower()

    # Series -> nullable Int8 is fine here
    extended_forecast_df["is_toyota"] = (maker_l == "toyota").astype("Int8")

    extended_forecast_df["is_due_mineral_synthetic"] = (
        (extended_forecast_df["is_due_mineral_oil"].astype("Int8") == 1) |
        (extended_forecast_df["is_due_synthetic_oil"].astype("Int8") == 1)
    ).astype("Int8")

    # warranty periods — NumPy booleans; cast with np.int8 (NOT "Int8")
    s5  = {"cadillac","lexus"}
    s6  = {"hyundai","kia","changan","renault","jetour","land rover"}
    s7  = {"mg","chery","haval"}
    s8  = {"geely"}

    car_age = extended_forecast_df["car_age"].to_numpy()
    in_s5 = maker_l.isin(s5).to_numpy()
    in_s6 = maker_l.isin(s6).to_numpy()
    in_s7 = maker_l.isin(s7).to_numpy()
    in_s8 = maker_l.isin(s8).to_numpy()
    in_other = ~(in_s5 | in_s6 | in_s7 | in_s8)

    w5 = in_s5 & (car_age >= 0) & (car_age < 5)
    w6 = in_s6 & (car_age >= 0) & (car_age < 6)
    w7 = in_s7 & (car_age >= 0) & (car_age < 7)
    w8 = in_s8 & (car_age >= 0) & (car_age < 8)
    w_other = in_other & (car_age >= 0) & (car_age < 4)

    extended_forecast_df["is_on_warranty_period"] = (w5 | w6 | w7 | w8 | w_other).astype(np.int8)

    # PMS final flag — NumPy cast to int8 (NOT "Int8")
    pms_last12 = pd.to_numeric(
        extended_forecast_df["total_number_of_pms_visits_last_12_months"],
        errors="coerce"
    ).fillna(0).to_numpy()
    brand_pms = {"geely","haval","hyundai","kia","mazda","mg","nissan","toyota"}
    in_brand_pms = maker_l.isin(brand_pms).to_numpy()

    extended_forecast_df["is_pms_final_flag"] = (
        (pms_last12 > 0) &
        (extended_forecast_df["is_on_warranty_period"].to_numpy() > 0) &
        in_brand_pms
    ).astype(np.int8)

    # free temps
    del maker_l, car_age, in_s5, in_s6, in_s7, in_s8, in_other, w5, w6, w7, w8, w_other, pms_last12, in_brand_pms
    gc.collect()

    logger.info("Filter final dataframe")
    columns_to_keep = [
        "is_pms_final_flag", "is_on_warranty_period",
        "_id","mobile","_observ_end_dt",
        # "PlateNumber",
        "customer_last_transaction_dt",
        "customer_days_since_last_trx",
        "is_due_mineral_oil",
        "is_due_synthetic_oil",
        "nationality",
        "maker",
        "model",
        "expected_number_of_visits_mineral_oil",
        "expected_number_of_visits_synthetic_oil",
        "is_due_PE",
        "is_due_PAC",
        "is_pac_city",
        "is_due",
        "preferred_language",
        "is_chinese_brand",
        "car_age",
        "loyalty_segment",
        "price_segment",
        "campaign_group",
        "decile",
        "is_station_closed",
        "churn_probability",
        "churn_bucket",
        "is_due_cabin_ac_filter","is_due_air_filter","is_due_fuel_additive",
        "is_due_brake_fluid","is_due_coolant","is_due_differential_oil",
        "is_due_fuel_filter","is_due_spark_plugs","is_due_atf",
        "is_due_cvt_oil","is_due_drive_belt","is_due_canister",
        "is_due_timing_belt","is_due_power_steering_pump_belt",
        "is_due_atf_filter","is_due_pms","is_due_tires",
        "is_due_engine_flush","is_due_battery","is_due_ac_services","is_due_mineral_oil",
        "is_due_synthetic_oil","is_due_headlight_polishing","is_due_suspension","is_due_PE","is_due_PAC","is_due_mineral_synthetic",
    ]
    extended_forecast_df = extended_forecast_df.loc[:, columns_to_keep]
    def drop_dupes_keep_most_data(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns
        name_to_indices = {}
        for i, name in enumerate(cols):
            name_to_indices.setdefault(name, []).append(i)

        keep_idx = []
        for name, idxs in name_to_indices.items():
            if len(idxs) == 1:
                keep_idx.append(idxs[0])
            else:
                # compute non-null counts per positional column
                nn_counts = [df.iloc[:, j].notna().sum() for j in idxs]
                # pick index of the max count; ties go to first occurrence
                best_local = int(np.argmax(nn_counts))
                keep_idx.append(idxs[best_local])

        # preserve left-to-right ordering
        keep_idx = sorted(keep_idx)
        return df.iloc[:, keep_idx]

    extended_forecast_df = drop_dupes_keep_most_data(extended_forecast_df)
    logger.info("Dropped duplicate columns, kept the one with most non-null values for each name.")
    logger.info("Reached Extended Forecast Filter - filtered columns")

    last_month = extended_forecast_df["_observ_end_dt"].max()
    filter_last = (extended_forecast_df["_observ_end_dt"] == last_month).to_numpy()
    logger.info("Created filter for last month based on max data")
    filtered_forecast_df = extended_forecast_df.loc[filter_last]
    del extended_forecast_df, filter_last
    gc.collect()
    logger.info("done all - at out variable")

    return [filtered_forecast_df]
