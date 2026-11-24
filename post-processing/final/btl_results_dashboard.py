"""Append a BTL wave's conversion results to the Mileage Model Dashboard workbook.

Re-run whenever the results window grows (e.g. extend --end by a few days). Each run:

  1. reads the wave's target list (TEST + CONTROL, with the mileage-model segments),
  2. pulls M-invoices for the window from MACDB (read-only SELECT, aggregated per invoice),
  3. counts an invoice as a conversion when its mobile OR its plate is on the list (TEST and CONTROL),
  4. aggregates target / converted / revenue / GP by
     campaign_group x price_segment x loyalty_segment x churn_bucket,
  5. writes to the dashboard:
       - Data:     replaces ONLY the rows whose 'Target Month' == --target-month (other months untouched)
       - <month>:  a dashboard sheet cloned from --template-sheet (same SUMIFS layout), if missing
       - Summary:  one row for the month, if missing
     A timestamped backup of the workbook is written first.

Definitions (match the existing Data sheet):
  target_customers  rows on the list (one row = one mobile x plate)
  total_converted   distinct M-invoices by listed customers in the window
  total_revenue     full invoice value: sum InvoiceBeforeVAT over ALL lines of those invoices
  total_gp          sum (InvoiceBeforeVAT - ItemCost)
  wave_2 columns    0 until a wave 2 is sent (pass --wave 2 to fill them)

Usage (BTL env):
    python post-processing/final/btl_results_dashboard.py \\
        --list "post-processing/final/outputs/nov-2026/nov_2026_wave_1_MASTER_2026-11-10.csv" \\
        --target-month "Nov 2026" --wave 1 --start 2026-11-10 --end 2026-11-23

Add --dry-run to print the numbers without touching the workbook.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd
import pyodbc
from copy import copy
from dotenv import load_dotenv

from btl_paths import OUTPUTS, REPO

DASHBOARD = Path("/home/ammar/OneDrive/automation-reports/mileage-model/Mileage Model Dashboard.xlsx")
BACKUP_DIR = OUTPUTS / "dashboard-backups"

DATA_COLS = [
    "Target Month", "campaign_group", "price_segment", "loyalty_segment", "churn_bucket",
    "target_customers", "total_converted", "converted_wave_1", "converted_wave_2",
    "total_revenue", "rev_wave_1", "rev_wave_2", "total_gp",
]
SEGMENTS = ["campaign_group", "price_segment", "loyalty_segment", "churn_bucket"]
# list spelling -> dashboard spelling
SEGMENT_FIXES = {"Uncommited": "Uncommitted", "Full Price": "Full price", "Promo Hunter": "Promo hunter"}


# ---------------------------------------------------------------- keys
def norm_mobile(s: pd.Series) -> pd.Series:
    """Any of 05XXXXXXXX / 5XXXXXXXX / 9665XXXXXXXX / 009665XXXXXXXX -> 9665XXXXXXXX."""
    d = s.astype("string").str.replace(r"\D", "", regex=True)
    d = d.str.replace(r"^00", "", regex=True)
    d = d.where(~d.str.match(r"^0\d{9}$", na=False), "966" + d.str[1:])
    d = d.where(~d.str.match(r"^5\d{8}$", na=False), "966" + d)
    return d.where(d.str.len() > 0)


def norm_plate(s: pd.Series) -> pd.Series:
    p = s.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    return p.where(~p.isin(["", "NAN", "NONE", "<NA>"]))


# ---------------------------------------------------------------- inputs
def load_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["mobile_key"] = norm_mobile(df["mobile"])
    df["plate_key"] = norm_plate(df["PlateNumber"])
    for c in SEGMENTS:
        df[c] = df[c].fillna("All").str.strip().replace(SEGMENT_FIXES)
    df["campaign_group"] = df["campaign_group"].str.upper()
    return df


def fetch_invoices(start: str, end_inclusive: str) -> pd.DataFrame:
    """M-invoices in [start, end] from MACDB — read-only SELECT, one row per invoice."""
    load_dotenv(REPO / ".env")
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.getenv('SERVER')};DATABASE={os.getenv('DATABASE')};"
        f"UID={os.getenv('USERNAME')};PWD={os.getenv('PASSWORD')};"
        "TrustServerCertificate=yes;ApplicationIntent=ReadOnly;",
        readonly=True,
    )
    end_excl = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sql = """
        SELECT s.InvoiceNumber,
               MIN(s.InvoiceDate)       AS InvoiceDate,
               SUM(s.InvoiceBeforeVAT)  AS revenue,
               SUM(s.ItemCost)          AS item_cost,
               MAX(c.Mobile)            AS Mobile,
               MAX(v.PlateNumber)       AS PlateNumber
        FROM dbo.MAC_Master_Sales_Pet AS s
            LEFT JOIN dbo.v_Customer  AS c ON s.CustomerID = c.CustomerID
            LEFT JOIN dbo.MAC_Vehicle AS v ON s.CustomerVehicleID = v.CustomerVehicleID
        WHERE s.InvoiceDate >= ? AND s.InvoiceDate < ?
          AND s.InvoiceNumber LIKE 'M%'
        GROUP BY s.InvoiceNumber
    """
    try:
        inv = pd.read_sql(sql, conn, params=[start, end_excl])
    finally:
        conn.close()
    inv["revenue"] = inv["revenue"].astype(float)
    inv["gp"] = inv["revenue"] - inv["item_cost"].astype(float)
    inv["mobile_key"] = norm_mobile(inv["Mobile"])
    inv["plate_key"] = norm_plate(inv["PlateNumber"])
    return inv


# ---------------------------------------------------------------- aggregation
def build_rows(lst: pd.DataFrame, inv: pd.DataFrame, target_month: str, wave: int) -> pd.DataFrame:
    # an invoice converts if its mobile OR its plate is on the list (TEST and CONTROL alike);
    # segments come from the mobile match first, else from the plate match
    by_mobile = lst.dropna(subset=["mobile_key"]).drop_duplicates("mobile_key")[["mobile_key", *SEGMENTS]]
    by_plate = lst.dropna(subset=["plate_key"]).drop_duplicates("plate_key")[["plate_key", *SEGMENTS]]
    m = inv.merge(by_mobile, on="mobile_key", how="inner").assign(match="mobile")
    p = inv[~inv["InvoiceNumber"].isin(m["InvoiceNumber"])].merge(by_plate, on="plate_key", how="inner")
    conv = pd.concat([m, p.assign(match="plate")], ignore_index=True).drop_duplicates("InvoiceNumber")
    print(f"matched invoices: {len(conv):,}  {conv['match'].value_counts().to_dict()}")
    conv["customer"] = conv["mobile_key"].fillna("") + "|" + conv["plate_key"].fillna("")

    target = lst.groupby(SEGMENTS).size().rename("target_customers")
    res = conv.groupby(SEGMENTS).agg(
        total_converted=("InvoiceNumber", "nunique"),
        converted_customers=("customer", "nunique"),
        total_revenue=("revenue", "sum"),
        total_gp=("gp", "sum"),
    )
    out = target.to_frame().join(res, how="left").fillna(0).reset_index()
    for c in ["target_customers", "total_converted", "converted_customers"]:
        out[c] = out[c].astype(int)
    out["total_revenue"] = out["total_revenue"].round(2)
    w1, w2 = (out["total_converted"], 0) if wave == 1 else (0, out["total_converted"])
    out["converted_wave_1"], out["converted_wave_2"] = w1, w2
    r1, r2 = (out["total_revenue"], 0) if wave == 1 else (0, out["total_revenue"])
    out["rev_wave_1"], out["rev_wave_2"] = r1, r2
    out.insert(0, "Target Month", target_month)
    order = {"TEST": 0, "CONTROL": 1}
    out = out.sort_values(SEGMENTS, key=lambda s: s.map(order) if s.name == "campaign_group" else s)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- workbook
def _copy_row_style(ws, src_row: int, dst_row: int, max_col: int) -> None:
    for col in range(1, max_col + 1):
        s, d = ws.cell(src_row, col), ws.cell(dst_row, col)
        if s.has_style:
            d._style = copy(s._style)


def write_data(wb, rows: pd.DataFrame, target_month: str) -> tuple[int, int]:
    ws = wb["Data"]
    header = [ws.cell(1, c).value for c in range(1, len(DATA_COLS) + 1)]
    assert header == DATA_COLS, f"Data header changed: {header}"
    last = max(r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value is not None)
    style_row = last
    # drop this month's rows only (from a previous run of this script), bottom-up
    mine = [r for r in range(2, last + 1) if ws.cell(r, 1).value == target_month]
    for r in reversed(mine):
        ws.delete_rows(r)
    last -= len(mine)
    style_row = min(style_row, last)
    for i, rec in enumerate(rows[DATA_COLS].itertuples(index=False), start=last + 1):
        for j, v in enumerate(rec, start=1):
            ws.cell(i, j, v.item() if hasattr(v, "item") else v)
        _copy_row_style(ws, style_row, i, len(DATA_COLS))
    return len(mine), len(rows)


def ensure_month_sheet(wb, target_month: str, template: str, note: str) -> str:
    if target_month in wb.sheetnames:
        ws = wb[target_month]
        action = "kept"
    else:
        src = wb[template]
        ws = wb.copy_worksheet(src)
        ws.title = target_month
        old, new = f"'{template}'!", f"'{target_month}'!"
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str) and c.value.startswith("="):
                    c.value = c.value.replace(old, new)
        ws.sheet_view.tabSelected = False
        wb.move_sheet(ws, offset=wb.sheetnames.index(template) + 1 - wb.sheetnames.index(target_month))
        action = "created"
    ws["I5"] = target_month          # the SUMIFS key into Data!A
    ws["I6"] = note
    ws["I6"].font = copy(ws["I5"].font)
    return action


def ensure_summary_row(wb, target_month: str) -> str:
    ws = wb["Summary"]
    rows = [r for r in range(4, ws.max_row + 1) if ws.cell(r, 4).value is not None]
    hit = [r for r in rows if ws.cell(r, 4).value == target_month]
    dst = hit[0] if hit else rows[-1] + 1
    src = dst - 1                       # formulas copied from the row above, shifted one row down

    def shift(m: re.Match) -> str:     # relative row refs only: D12 / $D12 -> D13 / $D13 (E$1 stays)
        col, dollar, row = m.groups()
        return f"{col}{dollar}{int(row) + 1}" if not dollar and int(row) == src else m.group(0)

    ws.cell(dst, 4, target_month)
    for col in range(4, ws.max_column + 1):
        s = ws.cell(src, col)
        if s.has_style:
            ws.cell(dst, col)._style = copy(s._style)
        if col > 4 and isinstance(s.value, str) and s.value.startswith("="):
            ws.cell(dst, col, re.sub(r"(\$?[A-Z]{1,3})(\$?)(\d+)\b", shift, s.value))
    ws.row_dimensions[dst].height = ws.row_dimensions[src].height
    return "refreshed" if hit else "added"


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, type=Path, help="wave target list CSV (TEST + CONTROL)")
    ap.add_argument("--target-month", required=True, help="label used in Data!A and the sheet name, e.g. 'Nov 2026'")
    ap.add_argument("--wave", type=int, default=1, choices=[1, 2])
    ap.add_argument("--start", required=True, help="first day of the results window (launch date)")
    ap.add_argument("--end", required=True, help="last day of the results window, inclusive")
    ap.add_argument("--workbook", type=Path, default=DASHBOARD)
    ap.add_argument("--template-sheet", default="Jul 2026")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lst = load_list(a.list)
    print(f"list: {len(lst):,} rows  {lst['campaign_group'].value_counts().to_dict()}")
    inv = fetch_invoices(a.start, a.end)
    print(f"MACDB: {len(inv):,} M-invoices {inv['InvoiceDate'].min()} .. {inv['InvoiceDate'].max()}")

    rows = build_rows(lst, inv, a.target_month, a.wave)
    tot = rows.groupby("campaign_group")[["target_customers", "total_converted", "converted_customers",
                                          "total_revenue", "total_gp"]].sum()
    tot["%convr"] = (tot["total_converted"] / tot["target_customers"]).round(4)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.2f}".format):
        print(tot)

    month_dir = OUTPUTS / pd.Timestamp(f"1 {a.target_month}").strftime("%b-%Y").lower()
    month_dir.mkdir(parents=True, exist_ok=True)
    detail = month_dir / f"results_wave_{a.wave}_{a.start}_to_{a.end}.csv"
    rows.to_csv(detail, index=False)
    print(f"rows: {len(rows)} -> {detail}")
    if a.dry_run:
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"{a.workbook.stem} {datetime.now():%Y-%m-%d %H%M%S}.xlsx"
    shutil.copy2(a.workbook, backup)
    print(f"backup: {backup}")

    wb = openpyxl.load_workbook(a.workbook)
    removed, added = write_data(wb, rows, a.target_month)
    note = f"Wave {a.wave} results {a.start} to {a.end} (M-invoices, updated {datetime.now():%Y-%m-%d})"
    sheet = ensure_month_sheet(wb, a.target_month, a.template_sheet, note)
    summ = ensure_summary_row(wb, a.target_month)
    wb.save(a.workbook)
    print(f"Data: -{removed} +{added} rows for '{a.target_month}' | sheet '{a.target_month}' {sheet} | Summary row {summ}")
    print(f"saved {a.workbook}")


if __name__ == "__main__":
    main()
