"""BTL target list — part 3: campaign names, MASTER list and the TEST workbook for marketing.

Port of `post-processing/rough/master_list.ipynb`. Input is the cleaned list from
btl_targetlist_clean.py (default: the last output recorded in outputs/exclusion_log.csv).

    python post-processing/final/btl_targetlist_master.py

Relabelling (what gets its own campaign)
  - preferred_language = AR when nationality is missing.
  - Due PMS            : churn / loyalty / price = All. Sub Product = PMS Hyundai / Kia / Nissan
                         by maker (overrides PAC / Chinese / Toyota / Standard PMS from part 1).
  - Due Non Oil        : churn / loyalty / price = All.
  - Due Oil + Non Oil  : churn / price = All (loyalty kept).
  - New Joiner         : churn = All.
  - Non-oil Sub Product collapses to two groups, first match wins:
        any of tires / battery / AC services / engine flush -> EF_BAT_ACS
        any of cabin filter / air filter / coolant / spark  -> ACF_AF_CO_SP
  - campaign_name = Product_SubProduct_churn_loyalty_price (empty tokens skipped),
    campaign_name_lang = campaign_name + _AR / _EN.

Outputs, in outputs/<mon-yyyy>/  (stem = input name without its date, e.g. nov_2026_wave_1)
  <stem>_MASTER_<date>.csv          every row (TEST + CONTROL) with campaign columns
  <stem>_TEST_list_<date>.xlsx      TEST only, one sheet per campaign_name_lang
  <stem>_test_grouped_<date>.csv    TEST customer counts per campaign

Differences vs the notebook (intentional)
  - Sheet names: Excel allows 31 chars. The notebook cut `campaign_name_lang[:31]`, which drops
    the _AR/_EN suffix on long names; the EN group was then written over the top of the AR sheet
    (first N AR customers lost, one mixed sheet). Now long names drop their "_All" tokens, then
    the campaign part is cut and the language suffix kept; duplicate sheet names raise instead of
    overwriting. Every row still carries the full campaign_name_lang.
  - The tires/battery/AC/flush group is named EF_BAT_ACS as the notebook's mapping intended
    (its key was in a different word order, so it fell back to the slug "is_due_tires_is_du").
    `Sub Product` keeps the notebook's long text for both groups.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from btl_paths import EXCLUSION_LOG

log = logging.getLogger("btl_targetlist_master")

EXCEL_SHEET_MAX = 31

PMS_MAKER_SUB_PRODUCT = {"HYUNDAI": "PMS Hyundai", "KIA": "PMS Kia", "NISSAN": "PMS Nissan"}

EF_BAT_ACS = "is_due_tires, is_due_battery, is_due_ac_services, is_due_engine_flush"
ACF_AF_CO_SP = "is_due_cabin_ac_filter, is_due_air_filter, is_due_coolant, is_due_spark_plugs"
NON_OIL_GROUPS = [  # (Sub Product written, pattern) — first match wins
    (EF_BAT_ACS, r"is_due_tires|is_due_battery|is_due_ac_services|is_due_engine_flush"),
    (ACF_AF_CO_SP, r"is_due_cabin_ac_filter|is_due_air_filter|is_due_coolant|is_due_spark_plugs"),
]

# Campaign-name tokens. Values not listed fall back to a slug of the value (as the notebook did),
# e.g. "Due PMS" -> "Due_PMS", "PAC PMS" -> "PAC_PMS", price "All" -> "All".
PRODUCT_TOKEN = {"Due Non Oil": "NonOil", "Due Oil": "Oil", "Due Oil + Non Oil": "Oil_NonOil"}
SUB_PRODUCT_TOKEN = {"Oil": "", EF_BAT_ACS: "EF_BAT_ACS", ACF_AF_CO_SP: "ACF_AF_CO_SP"}
CHURN_TOKEN = {"All": "", "High Probability": "Hi_Churn", "Low Probability": "Lo_Churn"}
LOYALTY_TOKEN = {"All": "", "Loyal": "Loyal", "New Joiner": "New", "Potential Loyal": "Pot_Loyal",
                 "Uncommited": "Uncom", "Lost n Found": "LnF"}
PRICE_TOKEN = {"Full price": "FP", "Mixed price": "MP", "Promo hunter": "Promo"}

GROUP_COLS = ["campaign_name", "Product", "Sub Product", "churn_bucket", "loyalty_segment", "price_segment"]


def slugify(value, max_len: int = 18) -> str:
    if pd.isna(value):
        return ""
    s = re.sub(r"[^\w]", "_", str(value).strip().replace(" ", "_"))
    return re.sub(r"_+", "_", s).strip("_")[:max_len]


def token(series: pd.Series, mapping: dict) -> pd.Series:
    """Map each value to its token; unmapped values get a slug (computed once per unique value)."""
    lookup = {v: mapping[v] if v in mapping else slugify(v) for v in series.dropna().unique()}
    return series.map(lookup).fillna("")


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    product = df["Product"]
    df.loc[df["nationality"].isna(), "preferred_language"] = "AR"
    df["maker"] = df["maker"].str.upper()

    pms = product == "Due PMS"
    df.loc[pms, ["churn_bucket", "loyalty_segment", "price_segment"]] = "All"
    by_maker = df["maker"].map(PMS_MAKER_SUB_PRODUCT)
    df.loc[pms & by_maker.notna(), "Sub Product"] = by_maker

    non_oil = product == "Due Non Oil"
    oil_non_oil = product == "Due Oil + Non Oil"
    df.loc[non_oil, ["churn_bucket", "loyalty_segment", "price_segment"]] = "All"
    df.loc[oil_non_oil, ["churn_bucket", "price_segment"]] = "All"

    has_non_oil = non_oil | oil_non_oil
    sub = df["Sub Product"].fillna("")
    group = np.select([sub.str.contains(p) for _, p in NON_OIL_GROUPS], [g for g, _ in NON_OIL_GROUPS], default="")
    df.loc[has_non_oil & (group != ""), "Sub Product"] = group[(has_non_oil & (group != "")).to_numpy()]

    df.loc[df["loyalty_segment"] == "New Joiner", "churn_bucket"] = "All"
    return df


def add_campaign_names(df: pd.DataFrame) -> pd.DataFrame:
    parts = pd.concat(
        [
            token(df["Product"], PRODUCT_TOKEN),
            token(df["Sub Product"], SUB_PRODUCT_TOKEN),
            token(df["churn_bucket"], CHURN_TOKEN),
            token(df["loyalty_segment"], LOYALTY_TOKEN),
            token(df["price_segment"], PRICE_TOKEN),
        ],
        axis=1,
    )
    # Join non-empty tokens with "_": few unique combinations, so do it on the uniques.
    combos = parts.drop_duplicates()
    names = combos.apply(lambda r: "_".join(t for t in r if t), axis=1)
    lookup = dict(zip(map(tuple, combos.to_numpy()), names))
    df["campaign_name"] = [lookup[t] for t in map(tuple, parts.to_numpy())]
    df["campaign_name_lang"] = df["campaign_name"] + "_" + df["preferred_language"].fillna("AR")
    return df


def sheet_names(names: list[str]) -> dict[str, str]:
    """campaign_name_lang -> unique Excel sheet name (<= 31 chars, language suffix kept)."""
    out = {}
    for name in names:
        if len(name) <= EXCEL_SHEET_MAX:
            out[name] = name
        else:
            base, _, lang = name.rpartition("_")
            base = base.replace("_All", "")  # "All" tokens carry no information
            out[name] = base[: EXCEL_SHEET_MAX - len(lang) - 1].rstrip("_") + "_" + lang
    clashes = pd.Series(out).loc[lambda s: s.duplicated(keep=False)]
    if len(clashes):
        raise ValueError(f"sheet names collide after shortening:\n{clashes.to_string()}")
    return out


def default_input() -> Path:
    if not EXCLUSION_LOG.exists():
        raise FileNotFoundError(f"{EXCLUSION_LOG} not found; pass --input")
    return Path(pd.read_csv(EXCLUSION_LOG)["output"].iloc[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, help="Cleaned list (default: last output in exclusion_log.csv).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    in_path = args.input or default_input()
    df = pd.read_csv(in_path, dtype={"mobile": str, "PlateNumber": str})
    log.info("loaded %s: %s rows", in_path, f"{len(df):,}")
    if df["mobile"].duplicated().any():
        raise ValueError("input has duplicate mobiles; it should be one row per mobile")

    df = add_campaign_names(relabel(df))

    out_dir = in_path.parent
    stem = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", in_path.stem)
    today = date.today().isoformat()

    master_path = out_dir / f"{stem}_MASTER_{today}.csv"
    df.to_csv(master_path, index=False)
    log.info("wrote %s (%s rows)", master_path.name, f"{len(df):,}")

    test = df[df["campaign_group"] == "TEST"]
    grouped = (
        test.groupby(GROUP_COLS, dropna=False)["_id"].nunique().rename("customers").reset_index()
        .sort_values(GROUP_COLS[1:])
    )
    grouped_path = out_dir / f"{stem}_test_grouped_{today}.csv"
    grouped.to_csv(grouped_path, index=False)
    log.info("wrote %s (%s campaigns, %s TEST customers)", grouped_path.name, len(grouped), f"{len(test):,}")

    sheets = sheet_names(sorted(test["campaign_name_lang"].unique()))
    xlsx_path = out_dir / f"{stem}_TEST_list_{today}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        for name, group in test.groupby("campaign_name_lang", sort=True):
            group.to_excel(writer, sheet_name=sheets[name], index=False)
    shortened = {k: v for k, v in sheets.items() if k != v}
    log.info("wrote %s (%s sheets, %s with shortened names)", xlsx_path.name, len(sheets), len(shortened))
    for full, short in shortened.items():
        log.info("  %-45s -> %s", full, short)


if __name__ == "__main__":
    main()
