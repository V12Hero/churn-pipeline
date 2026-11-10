"""BTL target list — part 2: remove customers who shouldn't be contacted this wave.

Port of `post-processing/rough/cleaning.ipynb`. Input is the CSV from btl_targetlist_create.py.

    python post-processing/final/btl_targetlist_clean.py \\
        --exclude "/path/to/jul_2026_wave_1_MASTER.csv" \\
        --exclude "/path/to/target_data_lost.xlsx#Mineral,Synthetic"

Rows whose mobile isn't a valid KSA mobile (e.g. the 966000000000 placeholder) are dropped first.
Then a row is dropped if its mobile OR its PlateNumber appears in any of:
  1. Recent sales  — MACDB invoices since --sales-since (default: the day after the latest
                     customer_last_transaction_dt in the list, i.e. every visit the model didn't see).
  2. No consent    — v_Customer.ConsentStatus = 0 (mobile only).
  3. --exclude     — lists already sent to marketing. CSV or XLSX; for XLSX append
                     `#Sheet1,Sheet2` to read only those sheets (default: every sheet).
                     Mobiles come from any `mobile` / `mapped_mobile` column (any case) or, failing
                     that, from `_id` ("<mobile>_<plate>"); plates from `PlateNumber`.

Every phone number (ours, MACDB's, the lists') is normalised to 12-digit `9665XXXXXXXX`, so
`0551234567`, `551234567`, `966551234567`, `+966 55 123 4567` and `00966551234567` all match.
In the notebook the consent filter compared int to str and silently removed nothing.

Every run appends one row per source to post-processing/final/outputs/exclusion_log.csv (full path,
sheets, counts, output file), so earlier exclusion lists can be found again. Paths are never
stored in this script: the lists change every wave.

Credentials: SERVER / DATABASE / USERNAME / PASSWORD from the repo `.env` (same as ingestion).
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from btl_paths import EXCLUSION_LOG, REPO, resolve_month_dir

log = logging.getLogger("btl_targetlist_clean")


MOBILE_COLUMNS = {"mobile", "mapped_mobile"}
PLATE_COLUMNS = {"platenumber"}

RECENT_SALES_SQL = """
    SELECT DISTINCT c.Mobile, v.PlateNumber
    FROM dbo.MAC_Master_Sales_Pet AS s
        LEFT JOIN dbo.v_Customer AS c ON s.CustomerID = c.CustomerID
        LEFT JOIN dbo.MAC_Vehicle AS v ON s.CustomerVehicleID = v.CustomerVehicleID
    WHERE s.InvoiceDate >= :since
"""
NO_CONSENT_SQL = "SELECT DISTINCT Mobile FROM dbo.v_Customer WHERE ConsentStatus = 0"


# ---------------------------------------------------------------- normalisation

def normalize_mobile(s: pd.Series) -> pd.Series:
    """Any KSA mobile spelling -> '9665XXXXXXXX'; anything else -> <NA>."""
    digits = s.astype("string").str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
    digits = digits.str.replace(r"^(00)?966", "", regex=True).str.lstrip("0")
    return ("966" + digits).where(digits.str.fullmatch(r"5\d{8}").fillna(False))


def normalize_plate(s: pd.Series) -> pd.Series:
    plate = s.astype("string").str.strip().str.upper()
    return plate.where(plate != "")


# ---------------------------------------------------------------- sources

def get_engine():
    load_dotenv(REPO / ".env")
    odbc = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.environ['SERVER']};DATABASE={os.environ['DATABASE']};"
        f"UID={os.environ['USERNAME']};PWD={os.environ['PASSWORD']};TrustServerCertificate=yes;"
    )
    return create_engine(URL.create("mssql+pyodbc", query={"odbc_connect": odbc}))


def recent_sales(engine, since: date) -> tuple[set, set]:
    df = pd.read_sql_query(text(RECENT_SALES_SQL), engine, params={"since": since.isoformat()})
    return set(normalize_mobile(df["Mobile"]).dropna()), set(normalize_plate(df["PlateNumber"]).dropna())


def no_consent(engine) -> set:
    df = pd.read_sql_query(text(NO_CONSENT_SQL), engine)
    return set(normalize_mobile(df["Mobile"]).dropna())


def read_list(spec: str) -> tuple[set, set]:
    """Mobiles and plates from one marketing list (`path` or `path.xlsx#Sheet1,Sheet2`)."""
    path_str, _, sheets = spec.partition("#")
    path = Path(path_str).expanduser()
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        frames = pd.read_excel(path, sheet_name=sheets.split(",") if sheets else None, dtype=str)
    else:
        frames = {path.name: pd.read_csv(path, dtype=str)}

    mobiles, plates = set(), set()
    for sheet, df in frames.items():
        cols = {c.strip().lower(): c for c in df.columns}
        mob_cols = [cols[c] for c in MOBILE_COLUMNS & cols.keys()]
        plate_cols = [cols[c] for c in PLATE_COLUMNS & cols.keys()]
        sheet_mobiles = set()
        for c in mob_cols:
            sheet_mobiles |= set(normalize_mobile(df[c]).dropna())
        if not mob_cols and "_id" in cols:
            sheet_mobiles = set(normalize_mobile(df[cols["_id"]].str.split("_").str[0]).dropna())
        sheet_plates = set()
        for c in plate_cols:
            sheet_plates |= set(normalize_plate(df[c]).dropna())
        if not sheet_mobiles and not sheet_plates:
            log.warning("  %s [%s]: no mobile/plate columns, skipped (columns: %s)", path.name, sheet, list(df.columns))
            continue
        log.info("  %s [%s]: %s rows, %s mobiles, %s plates", path.name, sheet,
                 f"{len(df):,}", f"{len(sheet_mobiles):,}", f"{len(sheet_plates):,}")
        mobiles |= sheet_mobiles
        plates |= sheet_plates
    return mobiles, plates


# ---------------------------------------------------------------- main

def drop(df: pd.DataFrame, name: str, mobiles: set, plates: set, report: list,
         path: str = "", sheets: str = "", extra: pd.Series | None = None) -> pd.DataFrame:
    hit = df["_mobile_norm"].isin(mobiles) | df["_plate_norm"].isin(plates)
    if extra is not None:
        hit |= extra
    report.append({"source": name, "path": path, "sheets": sheets, "mobiles": len(mobiles),
                   "plates": len(plates), "removed": int(hit.sum()), "left": int((~hit).sum())})
    log.info("%-40s removed %8s -> %s left", name, f"{hit.sum():,}", f"{(~hit).sum():,}")
    return df.loc[~hit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--month", help="e.g. 'Nov 2026' (default: newest folder in outputs/).")
    parser.add_argument("--input", type=Path,
                        help="Part-1 CSV (default: BTL_Targetlist_*.csv in the month folder).")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATH[#SHEETS]",
                        help="List already sent to marketing. Repeatable.")
    parser.add_argument("--sales-since", type=date.fromisoformat,
                        help="Drop customers with an invoice on/after this date "
                             "(default: day after the list's latest customer_last_transaction_dt).")
    parser.add_argument("--wave", default="1")
    parser.add_argument("--output-dir", type=Path, help="Default: the month folder.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = args.output_dir or resolve_month_dir(args.month)
    in_path = args.input or max(out_dir.glob("BTL_Targetlist_*.csv"), key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(in_path, dtype={"mobile": str, "PlateNumber": str})
    df["_mobile_norm"] = normalize_mobile(df["mobile"])
    df["_plate_norm"] = normalize_plate(df["PlateNumber"])
    log.info("loaded %s: %s rows", in_path, f"{len(df):,}")

    sales_since = args.sales_since or (
        pd.to_datetime(df["customer_last_transaction_dt"]).max().date() + pd.Timedelta(days=1)
    )
    report: list = []
    df = drop(df, "invalid mobile", set(), set(), report, extra=df["_mobile_norm"].isna())

    # Read every list before touching the DB, so a bad path fails fast.
    lists = []
    for spec in args.exclude:
        log.info("reading %s", spec)
        path_str, _, sheets = spec.partition("#")
        path = Path(path_str).expanduser().resolve()
        lists.append((path, sheets or "(all)", *read_list(spec)))

    engine = get_engine()
    log.info("querying MACDB: sales since %s, no-consent customers", sales_since)
    df = drop(df, f"recent sales (since {sales_since})", *recent_sales(engine, sales_since), report)
    df = drop(df, "no consent (ConsentStatus = 0)", no_consent(engine), set(), report)
    for path, sheets, mobiles, plates in lists:
        df = drop(df, path.name, mobiles, plates, report, str(path), sheets)

    target_month = df["Target_Month_Wave"].iloc[0] if len(df) else in_path.stem.removeprefix("BTL_Targetlist_")
    stem = f"{target_month.replace(' ', '_').lower()}_wave_{args.wave}_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.csv"
    df.drop(columns=["_mobile_norm", "_plate_norm"]).to_csv(out_path, index=False)

    summary = pd.DataFrame(report)
    log.info("summary:\n%s", summary.drop(columns=["path", "sheets"]).to_string(index=False))

    log_path = EXCLUSION_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary.insert(0, "run_at", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    summary.insert(1, "target_month", target_month)
    summary.insert(2, "wave", args.wave)
    summary["input"] = str(in_path.resolve())
    summary["output"] = str(out_path.resolve())
    summary.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    log.info("logged exclusions to %s", log_path)
    log.info("wrote %s (%s rows)", out_path, f"{len(df):,}")


if __name__ == "__main__":
    main()
