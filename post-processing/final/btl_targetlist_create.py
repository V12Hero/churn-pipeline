"""BTL target list — part 1: mileage forecast -> one row per mobile with the product to pitch.

Port of `post-processing/rough/BTL_Targetlist- Create.ipynb` (cells up to the first `to_csv`).
Not part of the Kedro pipeline; run it after `kedro run --pipeline full_mileage_model`.

    python post-processing/final/btl_targetlist_create.py
    python post-processing/final/btl_targetlist_create.py --as-of 2026-11-10
    python post-processing/final/btl_targetlist_create.py --recompute-price-segment
    python post-processing/final/btl_targetlist_create.py --search-engine "search-engine/11 Search Engine-Nov '26.xlsx"

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
  4. Closed stations: `last_branch_code` = branch of the customer's last visit (data/04_feature/
     ftr_branches at the forecast's _observ_end_dt); `is_station_closed` = 1 when that branch is in
     the ops Search Engine's "Permanent Closed Stations" sheet (incl. "Convert to PAC/NOS") and not
     back in "PE Stations List" (e.g. 1118 "Open Again"); `station_closing_date` from the same sheet.
     Uses the newest search-engine/*Search Engine-<Mon> '<yy>.xlsx (or --search-engine). This
     replaces the forecast's is_station_closed, which comes from mileage.closed_station_list (empty).

--recompute-price-segment
  Recomputes `price_segment` for the listed cars from data/03_primary/prm_transactions over the 12
  monthly snapshots ending at the forecast's _observ_end_dt (same window as the pipeline's
  d_feature_customer_transactions.create_segment_features). --price-rule picks what counts:
    pipeline           promo visit = a package on the promo list (catalog's raw_promos or --promos);
                       reproduces the Kedro rule: no promo visit -> Full price, all visits promo ->
                       Promo hunter, else Mixed price. `--promos <list the Kedro run used>` must
                       reproduce the forecast exactly.
    offers             promo visit also = a package named *OFFER* / *PROMO* or a 99/199/111/222
                       fixed-price oil package (partner deals, SAR 99 ...). Same three segments.
    offers+discount    (default) as `offers`, and Full price also needs no discount on any visit
                       (discount_amount > 0); a customer with discounted but no promo visits is Mixed.
  For the BTL tag only (user, 2026-11-10): the Kedro promo list and churn features keep the
  pipeline rule. Names match case/whitespace-insensitively.

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
import re
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from btl_paths import REPO, month_dir

log = logging.getLogger("btl_targetlist_create")

DEFAULT_INPUT = REPO / "data/06_models/mileage/mileage_forecast.parquet"
PRM_TRANSACTIONS = REPO / "data/03_primary/prm_transactions"
TRANSACTIONS_CATALOG = REPO / "conf/base/catalog/transactions_pipeline.yml"
FTR_BRANCHES = REPO / "data/04_feature/ftr_branches"
SEARCH_ENGINE_DIR = REPO / "search-engine"
PRICE_WINDOW_MONTHS = 12  # rowsBetween(-11, 0) over monthly snapshots in create_segment_features

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
    "last_branch_code",
    "is_station_closed",
    "station_closing_date",
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


def catalog_promos_path() -> Path:
    catalog = yaml.safe_load(TRANSACTIONS_CATALOG.read_text())
    return REPO / catalog["transactions.raw_promos"]["filepath"]


def _norm_names(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip().str.upper()


# Package names that count as a promo visit under --price-rule offers / offers+discount, on top of
# the promo list. The number must stand alone so model codes like "HILUX TGN111&121" don't match.
OFFER_NAME_PATTERN = r"OFFER|PROMO|(?<![A-Z0-9])(?:99|199|111|222)(?![0-9])"
PRICE_RULES = ["pipeline", "offers", "offers+discount"]


def recompute_price_segment(
    df: pd.DataFrame, target_month: str, promos_path: Path, rule: str = "offers+discount"
) -> pd.DataFrame:
    observ_end = pd.Timestamp(f"1 {target_month}") + pd.offsets.MonthEnd(0)
    window_start = (observ_end - pd.offsets.MonthEnd(PRICE_WINDOW_MONTHS - 1)).date()
    promos = set(_norm_names(pd.read_parquet(promos_path)["Promo"]).dropna())

    flt = (
        pc.field("_id").isin(df["_id"].unique().tolist())
        & (pc.field("_observ_end_dt") >= window_start)
        & (pc.field("_observ_end_dt") <= observ_end.date())
        & pc.field("transaction_id").is_valid()
    )
    tx = ds.dataset(PRM_TRANSACTIONS, format="parquet").to_table(
        filter=flt, columns=["_id", "transaction_id", "package_name", "discount_amount"]
    ).to_pandas()
    names = _norm_names(tx["package_name"])
    tx["is_promo"] = names.isin(promos).to_numpy()
    if rule != "pipeline":
        by_name = names.str.contains(OFFER_NAME_PATTERN, regex=True).fillna(False).to_numpy() & ~tx["is_promo"]
        log.info(
            "price rule %s: %s more promo lines by name, top:\n%s",
            rule, f"{by_name.sum():,}", names[by_name].value_counts().head(15).to_string(),
        )
        tx["is_promo"] |= by_name
    visits = tx.groupby("_id")["transaction_id"].nunique()
    promo_visits = tx.loc[tx["is_promo"]].groupby("_id")["transaction_id"].nunique()
    disc_visits = tx.loc[tx["discount_amount"].astype(float) > 0].groupby("_id")["transaction_id"].nunique()

    v = df["_id"].map(visits).fillna(0).to_numpy()
    pv = df["_id"].map(promo_visits).fillna(0).to_numpy()
    full = (v > 0) & (pv == 0)
    if rule == "offers+discount":
        full &= df["_id"].map(disc_visits).fillna(0).to_numpy() == 0
    new = np.select([full, (v > 0) & (pv == v)], ["Full price", "Promo hunter"], default="Mixed price")

    old = df["price_segment"].astype("string").fillna("<NA>").to_numpy()
    changes = pd.crosstab(pd.Series(old, name="forecast"), pd.Series(new, name="recomputed"))
    log.info(
        "price_segment recomputed (rule %s) with %s (%s promo names, snapshots %s..%s): %s of %s rows changed\n%s",
        rule, promos_path.name, len(promos), window_start, observ_end.date(),
        f"{(old != new).sum():,}", f"{len(df):,}", changes.to_string(),
    )
    df["price_segment"] = new
    return df


def latest_search_engine() -> Path:
    """Newest search-engine/*.xlsx by the "<Mon> '<yy>" in its name, e.g. "11 Search Engine-Nov '26.xlsx"."""
    dated = []
    for path in SEARCH_ENGINE_DIR.glob("*.xlsx"):
        m = re.search(r"([A-Za-z]{3})[a-z]* '(\d{2})", path.stem)
        if m and not path.name.startswith("~$"):
            dated.append((pd.to_datetime(f"1 {m[1]} 20{m[2]}", format="%d %b %Y"), path))
    if not dated:
        raise FileNotFoundError(f"no \"... Search Engine-<Mon> '<yy>.xlsx\" in {SEARCH_ENGINE_DIR}")
    return max(dated)[1]


def _branch_codes(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype("string")


def load_closed_stations(path: Path) -> pd.Series:
    """branch_code -> closing date for stations closed per the Search Engine workbook."""
    closed = pd.read_excel(path, sheet_name="Permanent Closed Stations", header=0)
    # The sheet has a summary block under the station rows; real rows have a 4-digit BE#.
    closed["branch_code"] = _branch_codes(closed["BE#"])
    closed = closed[closed["branch_code"].str.len().eq(4).fillna(False)]
    operational = set(_branch_codes(pd.read_excel(path, sheet_name="PE Stations List", header=1)["BE#"]).dropna())
    reopened = closed["branch_code"].isin(operational)
    closed = closed[~reopened]
    closed["closing_date"] = pd.to_datetime(closed["Closing Date"], errors="coerce").dt.date
    out = closed.sort_values("closing_date").drop_duplicates("branch_code", keep="last")
    log.info(
        "closed stations from %s: %s (%s back in PE Stations List ignored; status %s)",
        path.name, len(out), int(reopened.sum()), out["Current Status"].value_counts().to_dict(),
    )
    return out.set_index("branch_code")["closing_date"]


def flag_closed_station(df: pd.DataFrame, observ_end: pd.Timestamp, search_engine: Path) -> pd.DataFrame:
    flt = pc.field("_observ_end_dt") == observ_end.date()
    branches = ds.dataset(FTR_BRANCHES, format="parquet").to_table(
        filter=flt & pc.field("_id").isin(df["_id"].unique().tolist()), columns=["_id", "branch_code"]
    ).to_pandas()
    df["last_branch_code"] = df["_id"].map(branches.drop_duplicates("_id").set_index("_id")["branch_code"])
    df["last_branch_code"] = _branch_codes(df["last_branch_code"])

    closing = load_closed_stations(search_engine)
    df["station_closing_date"] = df["last_branch_code"].map(closing)
    df["is_station_closed"] = df["last_branch_code"].isin(closing.index).astype("int8")
    log.info(
        "last station closed: %s of %s rows (%s without a last branch)\n%s",
        f"{df['is_station_closed'].sum():,}", f"{len(df):,}", f"{df['last_branch_code'].isna().sum():,}",
        df.loc[df["is_station_closed"] == 1, "last_branch_code"].value_counts().head(15).to_string(),
    )
    return df


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
    parser.add_argument(
        "--recompute-price-segment",
        action="store_true",
        help="Recompute price_segment from prm_transactions with the current promo list.",
    )
    parser.add_argument(
        "--promos",
        type=Path,
        help="Promo list for --recompute-price-segment (default: the catalog's transactions.raw_promos).",
    )
    parser.add_argument(
        "--search-engine",
        type=Path,
        help="Ops Search Engine workbook for closed stations (default: newest in search-engine/).",
    )
    parser.add_argument(
        "--price-rule",
        choices=PRICE_RULES,
        default="offers+discount",
        help="What breaks Full price under --recompute-price-segment (see module docstring).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    target_month = args.month or target_month_from(args.input)
    log.info("target month %s, as-of %s", target_month, args.as_of)

    df = load_forecast(args.input)
    df = filter_active(df, args.as_of)
    df = assign_product(df)
    df = one_row_per_mobile(df)
    if args.recompute_price_segment:
        df = recompute_price_segment(df, target_month, args.promos or catalog_promos_path(), args.price_rule)
    observ_end = pd.Timestamp(f"1 {target_month}") + pd.offsets.MonthEnd(0)
    df = flag_closed_station(df, observ_end, args.search_engine or latest_search_engine())
    df = finalize(df, target_month)
    log_summary(df)

    out_dir = args.output_dir or month_dir(target_month)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"BTL_Targetlist_{target_month}.csv"
    df.to_csv(out_path, index=False)
    log.info("wrote %s (%s rows) at %s", out_path, f"{len(df):,}", datetime.now().strftime("%H:%M"))


if __name__ == "__main__":
    main()
