"""BTL target list — part 1: mileage forecast -> one row per mobile with the product to pitch.

Port of `post-processing/rough/BTL_Targetlist- Create.ipynb` (cells up to the first `to_csv`).
Not part of the Kedro pipeline; run it after `kedro run --pipeline full_mileage_model`.

    python post-processing/final/btl_targetlist_create.py
    python post-processing/final/btl_targetlist_create.py --as-of 2026-11-10

Steps
  1. Keep active customers: loyalty_segment not Lost/Gone, last transaction within 365 days of --as-of.
  2. Give every car a product, highest priority first:
       Due PMS      is_due_pms & is_due_mineral_synthetic  -> sub product: PAC / Chinese / Toyota / Standard PMS
       Due Oil      mineral/synthetic due, no non-oil due  -> sub product: "Oil"
       Due Oil + Non Oil   mineral/synthetic due + non-oil -> sub product: the non-oil list
       Due Non Oil  only non-oil due                       -> sub product: the non-oil list
     Cars with none of these are dropped.
  3. One row per mobile: best product across the customer's cars; ties go to the car with the
     latest transaction.

Differences vs the notebook (intentional)
  - The 365-day cutoff is anchored to --as-of (default: today) so a rerun is reproducible.
  - The `transaction_year >= 2025` and `due_anything > 0` filters are dropped: both are implied
    by the 365-day cutoff and the product assignment respectively.
  - The notebook's `sub_product_Category` (two hand-kept lists of exact combinations) is dropped:
    nothing downstream used it; btl_targetlist_master.py groups non-oil by a simple rule instead.
  - Within a mobile, the car with the latest transaction wins (notebook: whichever row came first).
  - `churn_bucket` is overwritten with a 2-level High/Low label (churn_probability >= 0.75), as in
    the notebook; downstream summaries group on that name.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from btl_paths import REPO, month_dir

log = logging.getLogger("btl_targetlist_create")

DEFAULT_INPUT = REPO / "data/06_models/mileage/mileage_forecast.parquet"

RECENCY_DAYS = 365
EXCLUDED_SEGMENTS = ["Lost", "Gone"]
CHURN_HIGH_THRESHOLD = 0.75

# Non-oil services that make up `Sub Product` for the non-oil products, comma-joined in this order.
# btl_targetlist_master.py later collapses them into the EF_BAT_ACS / ACF_AF_CO_SP campaign groups.
NON_OIL_DUE_COLUMNS = [
    "is_due_cabin_ac_filter",
    "is_due_air_filter",
    "is_due_coolant",
    "is_due_spark_plugs",
    "is_due_tires",
    "is_due_engine_flush",
    "is_due_battery",
    "is_due_ac_services",
]

# Product priority when a customer qualifies for several (lower = wins).
PRODUCT_PRIORITY = {"Due PMS": 0, "Due Oil": 1, "Due Oil + Non Oil": 1, "Due Non Oil": 2}

PASSTHROUGH_COLUMNS = [
    "mobile",
    "customer_last_transaction_dt",
    "nationality",
    "maker",
    "model",
    "preferred_language",
    "campaign_group",
    "car_age",
    "loyalty_segment",
    "price_segment",
    "decile",
    "is_station_closed",
    "churn_probability",
]
READ_COLUMNS = (
    ["_id", "is_due_pms", "is_due_mineral_synthetic", "is_pac_city", "is_chinese_brand"]
    + PASSTHROUGH_COLUMNS
    + NON_OIL_DUE_COLUMNS
)
OUTPUT_COLUMNS = [
    "_id",
    "mobile",
    "PlateNumber",
    "customer_last_transaction_dt",
    "nationality",
    "maker",
    "model",
    "preferred_language",
    "campaign_group",
    "car_age",
    "loyalty_segment",
    "price_segment",
    "decile",
    "is_station_closed",
    "churn_probability",
    "churn_bucket",
    "Product",
    "Sub Product",
    "Target_Month_Wave",
]


def load_forecast(path: Path) -> pd.DataFrame:
    # Read only the columns we use; ignore pandas metadata (the categorical/period extension
    # types written by Kedro don't always round-trip).
    table = pq.read_table(path, columns=READ_COLUMNS)
    df = table.to_pandas(ignore_metadata=True)
    log.info("loaded %s: %s rows", path, f"{len(df):,}")
    return df


def filter_active(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=RECENCY_DAYS)
    keep = ~df["loyalty_segment"].isin(EXCLUDED_SEGMENTS) & (
        df["customer_last_transaction_dt"] > cutoff
    )
    out = df.loc[keep].copy()
    log.info(
        "active (segment not %s, last trx after %s): %s rows",
        "/".join(EXCLUDED_SEGMENTS),
        cutoff.date(),
        f"{len(out):,}",
    )
    return out


def build_non_oil_due(df: pd.DataFrame) -> pd.Series:
    """Comma-joined names of the due non-oil services, in NON_OIL_DUE_COLUMNS order ('' if none)."""
    flags = df[NON_OIL_DUE_COLUMNS].to_numpy() == 1
    names = np.array([c + ", " for c in NON_OIL_DUE_COLUMNS], dtype=object)
    joined = (flags * names).sum(axis=1)  # object '' * False -> '', so this concatenates
    return pd.Series(joined, index=df.index, dtype=object).str.removesuffix(", ")


def assign_product(df: pd.DataFrame) -> pd.DataFrame:
    non_oil = build_non_oil_due(df)
    has_non_oil = (non_oil != "").to_numpy()
    oil = (df["is_due_mineral_synthetic"] == 1).to_numpy()
    pms = oil & (df["is_due_pms"] == 1).to_numpy()

    product = np.select(
        [pms, oil & ~has_non_oil, oil & has_non_oil, has_non_oil],
        ["Due PMS", "Due Oil", "Due Oil + Non Oil", "Due Non Oil"],
        default="",
    )
    pms_type = np.select(
        [(df["is_pac_city"] == 1).to_numpy(), (df["is_chinese_brand"] == 1).to_numpy(), (df["maker"] == "TOYOTA").to_numpy()],
        ["PAC PMS", "Chinese PMS", "Toyota PMS"],
        default="Standard PMS",
    )
    sub_product = np.select(
        [product == "Due PMS", product == "Due Oil"],
        [pms_type, "Oil"],
        default=non_oil.to_numpy(),
    )

    df["Product"] = product
    df["Sub Product"] = sub_product
    out = df.loc[product != ""].copy()
    log.info("with a product to pitch: %s rows", f"{len(out):,}")
    return out


def one_row_per_mobile(df: pd.DataFrame) -> pd.DataFrame:
    df["_priority"] = df["Product"].map(PRODUCT_PRIORITY)
    df = df.sort_values(
        ["mobile", "_priority", "customer_last_transaction_dt"],
        ascending=[True, True, False],
        kind="stable",
    )
    out = df.drop_duplicates(subset="mobile", keep="first").drop(columns="_priority")
    log.info("one row per mobile: %s rows", f"{len(out):,}")
    return out


def finalize(df: pd.DataFrame, target_month: str) -> pd.DataFrame:
    df["mobile"] = df["mobile"].astype(str)
    df["PlateNumber"] = df["_id"].str.split("__").str[0]
    df["_id"] = df["mobile"] + "_" + df["PlateNumber"]
    df["churn_bucket"] = np.where(
        df["churn_probability"] >= CHURN_HIGH_THRESHOLD, "High Probability", "Low Probability"
    )
    df["campaign_group"] = df["campaign_group"].str.upper()
    df["Target_Month_Wave"] = target_month
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def log_summary(df: pd.DataFrame) -> None:
    log.info("Product:\n%s", df["Product"].value_counts().to_string())


def target_month_from(df_path: Path) -> str:
    """'Nov 2026' from the forecast's single _observ_end_dt."""
    dates = pq.read_table(df_path, columns=["_observ_end_dt"]).column(0).unique().to_pylist()
    if len(dates) != 1:
        raise ValueError(f"expected one _observ_end_dt in {df_path}, found {dates}")
    return pd.Timestamp(dates[0]).strftime("%b %Y")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: post-processing/final/outputs/<mon-yyyy>/")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="Date the 365-day recency cutoff counts back from (default: today).",
    )
    parser.add_argument(
        "--month",
        help="Target_Month_Wave label, e.g. 'Nov 2026' (default: from the forecast's _observ_end_dt).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    target_month = args.month or target_month_from(args.input)
    log.info("target month %s, as-of %s", target_month, args.as_of)

    df = load_forecast(args.input)
    df = filter_active(df, args.as_of)
    df = assign_product(df)
    df = one_row_per_mobile(df)
    df = finalize(df, target_month)
    log_summary(df)

    out_dir = args.output_dir or month_dir(target_month)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"BTL_Targetlist_{target_month}.csv"
    df.to_csv(out_path, index=False)
    log.info("wrote %s (%s rows) at %s", out_path, f"{len(df):,}", datetime.now().strftime("%H:%M"))


if __name__ == "__main__":
    main()
