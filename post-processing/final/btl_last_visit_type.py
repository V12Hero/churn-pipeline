"""BTL target list — add-on: split the sent TEST list by the type of the customer's last visit.

Quick one-off on top of part 3 (no Kedro run). Input is the MASTER list from btl_targetlist_master.py;
only TEST rows (what marketing got) are kept.

    python post-processing/final/btl_last_visit_type.py
    python post-processing/final/btl_last_visit_type.py --month "Nov 2026" --sales-since 2026-10-19

Last visit = the latest invoice in data/03_primary/prm_transactions, matched on the car (plate +
mobile); if the car has no invoice there, the mobile's latest invoice; then the plate's.
Visit type of that invoice:
    Synthetic  any line with major_category = oil_synthetic
    Mineral    else any line with major_category = oil
    Non Oil    everything else
Rows are dropped when the mobile or plate has a MACDB invoice on/after --sales-since (default: day
after the list's latest customer_last_transaction_dt), i.e. they converted after the list was built.

Output: <month folder>/<stem>_last_visit_type_<date>.xlsx, sheets Synthetic / Mineral / Non Oil,
without campaign or due-product columns.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from btl_paths import REPO, resolve_month_dir
from btl_targetlist_clean import get_engine, normalize_mobile, normalize_plate, recent_sales

log = logging.getLogger("btl_last_visit_type")

PRM_TRANSACTIONS = REPO / "data/03_primary/prm_transactions"
DROP_COLUMNS = ["Product", "Sub Product", "campaign_name", "campaign_name_lang", "Target_Month_Wave"]
SHEETS = ["Synthetic", "Mineral", "Non Oil"]


def last_invoices() -> pd.DataFrame:
    """One row per invoice: _id, plate, mobile, date, visit type."""
    tbl = ds.dataset(PRM_TRANSACTIONS, format="parquet").to_table(
        columns=["_id", "transaction_id", "transaction_dt", "major_category"],
        filter=pc.field("transaction_id").is_valid(),
    )
    tx = tbl.to_pandas()
    log.info("prm_transactions: %s lines, last date %s", f"{len(tx):,}", tx["transaction_dt"].max())
    tx["syn"] = tx["major_category"].eq("oil_synthetic")
    tx["min"] = tx["major_category"].eq("oil")
    inv = tx.groupby("transaction_id", sort=False).agg(
        _id=("_id", "first"), dt=("transaction_dt", "max"), syn=("syn", "any"), min=("min", "any")
    )
    del tx
    inv["visit_type"] = np.select([inv["syn"], inv["min"]], ["Synthetic", "Mineral"], default="Non Oil")
    parts = inv["_id"].str.split("__", n=1, expand=True)
    inv["plate"] = normalize_plate(parts[0])
    inv["mobile"] = normalize_mobile(parts[1])
    inv = inv.sort_values("dt", ascending=False, kind="stable")
    log.info("invoices: %s", f"{len(inv):,}")
    return inv[["plate", "mobile", "dt", "visit_type"]].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--month", help="e.g. 'Nov 2026' (default: newest folder in outputs/).")
    parser.add_argument("--input", type=Path, help="MASTER CSV (default: newest *_MASTER_*.csv in the month folder).")
    parser.add_argument("--sales-since", type=date.fromisoformat)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = resolve_month_dir(args.month)
    in_path = args.input or max(out_dir.glob("*_MASTER_*.csv"), key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(in_path, dtype=str)
    df = df[df["campaign_group"].str.upper() == "TEST"].copy()
    df["_m"] = normalize_mobile(df["mobile"])
    df["_p"] = normalize_plate(df["PlateNumber"])
    log.info("loaded %s: %s TEST rows", in_path, f"{len(df):,}")

    inv = last_invoices()
    by_car = inv.drop_duplicates(["plate", "mobile"]).set_index(["plate", "mobile"])
    by_mobile = inv.dropna(subset=["mobile"]).drop_duplicates("mobile").set_index("mobile")
    by_plate = inv.dropna(subset=["plate"]).drop_duplicates("plate").set_index("plate")

    car_key = pd.MultiIndex.from_arrays([df["_p"], df["_m"]])
    for col in ["visit_type", "dt"]:
        car = pd.Series(by_car[col].reindex(car_key).to_numpy(), index=df.index)
        mob = df["_m"].map(by_mobile[col])
        plt = df["_p"].map(by_plate[col])
        df[f"last_{col}"] = car.fillna(mob).fillna(plt)
    matched = np.select(
        [pd.Series(by_car["visit_type"].reindex(car_key).to_numpy()).notna().to_numpy(),
         df["_m"].isin(by_mobile.index).to_numpy(), df["_p"].isin(by_plate.index).to_numpy()],
        ["car", "mobile", "plate"], default="none",
    )
    log.info("matched on: %s", pd.Series(matched).value_counts().to_dict())
    df["last_visit_type"] = df["last_visit_type"].fillna("Non Oil")

    sales_since = args.sales_since or (pd.to_datetime(df["customer_last_transaction_dt"]).max().date()
                                       + pd.Timedelta(days=1))
    mobiles, plates = recent_sales(get_engine(), sales_since)
    hit = df["_m"].isin(mobiles) | df["_p"].isin(plates)
    log.info("recent sales since %s: removed %s -> %s left", sales_since, f"{hit.sum():,}", f"{(~hit).sum():,}")
    df = df.loc[~hit]

    # No mobile or plate may appear twice anywhere in the workbook: keep the most recent visit.
    n = len(df)
    df = df.sort_values("last_dt", ascending=False, na_position="last", kind="stable")
    df = df[~(df["_m"].duplicated() & df["_m"].notna())]
    df = df[~(df["_p"].duplicated() & df["_p"].notna())]
    assert not df["_m"].dropna().duplicated().any() and not df["_p"].dropna().duplicated().any()
    log.info("dedupe mobile/plate: removed %s -> %s left", f"{n - len(df):,}", f"{len(df):,}")

    df = df.rename(columns={"last_dt": "last_visit_date"})
    df["last_visit_date"] = pd.to_datetime(df["last_visit_date"]).dt.date
    df = df.drop(columns=[c for c in DROP_COLUMNS + ["_m", "_p"] if c in df.columns])
    log.info("last visit type:\n%s", df["last_visit_type"].value_counts().to_string())

    stem = in_path.stem.split("_MASTER_")[0]
    out_path = out_dir / f"{stem}_last_visit_type_{date.today().isoformat()}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for sheet in SHEETS:
            df[df["last_visit_type"] == sheet].to_excel(xw, sheet_name=sheet, index=False)
    log.info("wrote %s (%s rows)", out_path, f"{len(df):,}")


if __name__ == "__main__":
    main()
