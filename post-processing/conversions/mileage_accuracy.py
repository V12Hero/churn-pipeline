"""How far off was the predicted mileage from the odometer the converted cars came in with?

For a BTL wave:
  1. converted cars = M-invoices in the window whose PLATE is on the wave list
     (match_type "plate + mobile" when the invoice mobile is the listed one too, else "plate only";
      mobile-only matches are left out on purpose: we're judging the car, not the customer),
  2. first visit per car in the window only (repeat visits ignored),
  3. actual   = WorkOrderMileage on that invoice (MACDB, read-only, M-invoices only),
     predicted = mileage model features (ftr_mileage) at the forecast snapshot:
       - predicted_at_visit: last known odometer + avg km/day x days from last visit to THIS visit
       - predicted_month_end: the model's own customer_mileage_forecast (projected to snapshot month-end)
  4. writes a formatted Excel: README, Summary, By segment, Error bands, Detail.

Usage (BTL env, from the repo root):
    python post-processing/conversions/mileage_accuracy.py \\
        --list "post-processing/final/outputs/nov-2026/nov_2026_wave_1_MASTER_2026-11-10.csv" \\
        --target-month "Nov 2026" --wave 1 --start 2026-11-10 --end 2026-11-23

Output: post-processing/conversions/outputs/<mon-yyyy>/mileage_accuracy_wave_<n>_<start>_to_<end>.xlsx
(git-ignored: it holds mobiles and plates).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyodbc
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "final"))
from btl_results_dashboard import load_list, norm_mobile, norm_plate  # noqa: E402

FTR_MILEAGE = REPO / "data/04_feature/ftr_mileage"
OUTPUTS = HERE / "outputs"

DEFAULT_KM_PER_DAY = 55   # the model's fallback rate when a car has no usable km/day history
KM_BANDS = [-np.inf, 2500, 5000, 7500, 10000, 15000, np.inf]
KM_BAND_LABELS = ["< 2.5k", "2.5k-5k", "5k-7.5k", "7.5k-10k", "10k-15k", "15k+"]
PRODUCT_ORDER = ["Due PMS", "Due Oil", "Due Oil + Non Oil", "Due Non Oil"]
VISIT_ORDER = ["PMS", "Oil", "Non Oil only"]
MAX_KM_PER_DAY = 700   # same cap the mileage model applies to customer_mileage_per_day
BANDS = [-np.inf, -10000, -5000, -2500, -1000, 1000, 2500, 5000, 10000, np.inf]
BAND_LABELS = ["< -10k (came much lower)", "-10k to -5k", "-5k to -2.5k", "-2.5k to -1k", "within ±1k",
               "+1k to +2.5k", "+2.5k to +5k", "+5k to +10k", "> +10k (came much higher)"]


# ---------------------------------------------------------------- data
def fetch_invoices(start: str, end_inclusive: str) -> pd.DataFrame:
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
               MAX(s.WorkOrderMileage)  AS actual_mileage,
               MAX(CASE WHEN s.ItemCategory = 'Oil' THEN 1 ELSE 0 END) AS has_oil,
               MAX(CASE WHEN s.IsPMS = 1 THEN 1 ELSE 0 END)            AS is_pms,
               SUM(s.InvoiceBeforeVAT)  AS revenue,
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
    inv["InvoiceDate"] = pd.to_datetime(inv["InvoiceDate"])
    inv["actual_mileage"] = pd.to_numeric(inv["actual_mileage"], errors="coerce")
    inv["mobile_key"] = norm_mobile(inv["Mobile"])
    inv["plate_key"] = norm_plate(inv["PlateNumber"])
    inv["actual_visit"] = np.select([inv["is_pms"] == 1, inv["has_oil"] == 1], ["PMS", "Oil"], "Non Oil only")
    return inv


def first_visits(lst: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """Plate-matched invoices -> one row per car: its first invoice in the window."""
    keep = ["mobile_key", "plate_key", "mobile", "PlateNumber", "campaign_group", "loyalty_segment",
            "price_segment", "churn_bucket", "Product", "maker", "model", "car_age", "customer_last_transaction_dt"]
    listed = lst.dropna(subset=["plate_key"])[[c for c in keep if c in lst.columns]]
    m = inv.dropna(subset=["plate_key"]).merge(listed, on="plate_key", how="inner", suffixes=("", "_list"))
    m["match_type"] = np.where(m["mobile_key"] == m["mobile_key_list"], "plate + mobile", "plate only")
    # a plate listed under several mobiles: prefer the listed row whose mobile is the invoice mobile
    m = m.sort_values(["InvoiceNumber", "match_type"]).drop_duplicates("InvoiceNumber")
    m = m.sort_values(["plate_key", "InvoiceDate", "InvoiceNumber"])
    m["visits_in_window"] = m.groupby("plate_key")["InvoiceNumber"].transform("size")
    return m.drop_duplicates("plate_key", keep="first").reset_index(drop=True)


def load_predictions(snapshot: pd.Timestamp) -> pd.DataFrame:
    cols = ["_id", "customer_mileage_forecast", "customer_last_current_mileage",
            "customer_avg_mileage_per_day", "customer_last_transaction_dt"]
    t = ds.dataset(FTR_MILEAGE, format="parquet").to_table(
        columns=cols, filter=pc.field("_observ_end_dt") == snapshot.date())
    p = t.to_pandas()
    parts = p["_id"].str.split("__", n=1, expand=True)        # ftr _id = <plate>__<mobile>
    p["plate_key"], p["mobile_key"] = norm_plate(parts[0]), norm_mobile(parts[1])
    return p.drop(columns="_id").rename(columns={"customer_last_transaction_dt": "model_last_visit_dt"})


def compare(fv: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    # the list row's (plate, mobile) is the model's _id
    df = fv.merge(pred, left_on=["plate_key", "mobile_key_list"], right_on=["plate_key", "mobile_key"],
                  how="left", suffixes=("", "_pred")).drop(columns=["mobile_key_pred"])
    days = (df["InvoiceDate"].dt.normalize() - pd.to_datetime(df["model_last_visit_dt"]).dt.normalize()).dt.days
    df["days_since_model_last_visit"] = days
    df["predicted_at_visit"] = (df["customer_last_current_mileage"]
                                + df["customer_avg_mileage_per_day"] * days).round()
    df["predicted_month_end"] = df["customer_mileage_forecast"].round()
    df["error_km"] = df["actual_mileage"] - df["predicted_at_visit"]          # + = came with MORE km
    df["abs_error_km"] = df["error_km"].abs()
    df["error_pct"] = df["error_km"] / df["predicted_at_visit"]
    df["abs_error_pct"] = df["error_pct"].abs()
    df["error_vs_month_end_km"] = df["actual_mileage"] - df["predicted_month_end"]
    df["actual_km_per_day"] = ((df["actual_mileage"] - df["customer_last_current_mileage"]) / days).round(1)
    df["predicted_km_since_last"] = df["predicted_at_visit"] - df["customer_last_current_mileage"]
    df["actual_km_since_last"] = df["actual_mileage"] - df["customer_last_current_mileage"]
    df["rate_source"] = np.where(df["customer_avg_mileage_per_day"] == DEFAULT_KM_PER_DAY,
                                 f"default {DEFAULT_KM_PER_DAY} km/day (no usable history)", "car's own history")

    status = np.select(
        [df["predicted_at_visit"].isna(),
         df["actual_mileage"].isna() | (df["actual_mileage"] <= 0),
         df["actual_mileage"] < df["customer_last_current_mileage"],
         df["actual_km_per_day"] > MAX_KM_PER_DAY],
        ["no prediction (no odometer history)", "no odometer on invoice", "odometer below last reading",
         f"odometer implausibly high (> {MAX_KM_PER_DAY} km/day since last visit)"],
        default="ok")
    df["status"] = status
    df["error_band"] = pd.cut(df["error_km"], BANDS, labels=BAND_LABELS)
    df.loc[df["status"] != "ok", "error_band"] = np.nan
    return df


# ---------------------------------------------------------------- summaries
def stats(g: pd.DataFrame) -> pd.Series:
    ok = g[g["status"] == "ok"]
    e, a = ok["error_km"], ok["abs_error_km"]
    return pd.Series({
        "cars": len(g),
        "comparable cars": len(ok),
        "median error km (bias)": e.median(),
        "mean error km (bias)": e.mean(),
        "median abs error km": a.median(),
        "mean abs error km": a.mean(),
        "median abs error %": ok["abs_error_pct"].median(),
        "within ±1,000 km": (a <= 1000).mean() if len(ok) else np.nan,
        "within ±2,500 km": (a <= 2500).mean() if len(ok) else np.nan,
        "within ±5,000 km": (a <= 5000).mean() if len(ok) else np.nan,
        "came higher than predicted": (e > 0).mean() if len(ok) else np.nan,
        "median predicted km/day": ok["customer_avg_mileage_per_day"].median(),
        "median actual km/day": ok["actual_km_per_day"].median(),
        "median error vs month-end forecast km": ok["error_vs_month_end_km"].median(),
    })


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overall = pd.concat([stats(df).rename("All converted cars")]
                        + [stats(g).rename(k) for k, g in df.groupby("match_type")]
                        + [stats(g).rename(k) for k, g in df.groupby("campaign_group")], axis=1)
    status = df["status"].value_counts().rename_axis("status").reset_index(name="cars")

    seg_frames = []
    for col, title in [("campaign_group", "Campaign group"), ("loyalty_segment", "Loyalty segment"),
                       ("price_segment", "Price segment"), ("churn_bucket", "Churn bucket"),
                       ("Product", "Product"), ("match_type", "Match type"),
                       ("rate_source", "Predicted km/day source")]:
        t = df.groupby(col, dropna=False).apply(stats, include_groups=False).reset_index()
        t.insert(0, "dimension", title)
        seg_frames.append(t.rename(columns={col: "segment"}))
    km_bins = [0, 50000, 100000, 150000, 200000, 300000, np.inf]
    km_lab = ["< 50k", "50k-100k", "100k-150k", "150k-200k", "200k-300k", "300k+"]
    for col, title, bins, lab in [
        ("customer_last_current_mileage", "Last known odometer", km_bins, km_lab),
        ("days_since_model_last_visit", "Days since last visit", [0, 90, 180, 365, 730, np.inf],
         ["< 3 months", "3-6 months", "6-12 months", "1-2 years", "2+ years"]),
    ]:
        t = df.groupby(pd.cut(df[col], bins, labels=lab), observed=False).apply(stats, include_groups=False).reset_index()
        t.insert(0, "dimension", title)
        seg_frames.append(t.rename(columns={col: "segment"}))
    by_seg = pd.concat(seg_frames, ignore_index=True)

    ok = df[df["status"] == "ok"]
    bands = ok["error_band"].value_counts().reindex(BAND_LABELS, fill_value=0).rename_axis("error band").reset_index(name="cars")
    bands["share"] = bands["cars"] / bands["cars"].sum()
    for grp in ["TEST", "CONTROL"]:
        bands[grp] = ok.loc[ok["campaign_group"] == grp, "error_band"].value_counts().reindex(BAND_LABELS, fill_value=0).values
    return {"overall": overall, "status": status, "by_segment": by_seg, "bands": bands, **confusion(df)}


def confusion(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # 1) what the list said the car was due for  vs  what it came for (all converted cars)
    prod = pd.crosstab(df["Product"], df["actual_visit"]).reindex(index=PRODUCT_ORDER, columns=VISIT_ORDER, fill_value=0)
    prod.index.name = "Predicted (list Product)"

    # 2) oil due (Due PMS / Due Oil / Due Oil + Non Oil) vs oil actually on the invoice
    pred_oil = np.where(df["Product"].eq("Due Non Oil"), "Predicted: NOT due for oil", "Predicted: due for oil")
    act_oil = np.where(df["has_oil"] == 1, "Actual: bought oil", "Actual: no oil")
    oil = pd.crosstab(pred_oil, act_oil).reindex(index=["Predicted: due for oil", "Predicted: NOT due for oil"],
                                                 columns=["Actual: bought oil", "Actual: no oil"], fill_value=0)
    oil.index.name = "Predicted"
    tp, fp = oil.iloc[0, 0], oil.iloc[0, 1]
    fn, tn = oil.iloc[1, 0], oil.iloc[1, 1]
    oil_metrics = pd.DataFrame({"metric": [
        "Precision: of cars predicted due for oil, share that bought oil",
        "Recall: of cars that bought oil, share predicted due for oil",
        "Accuracy: share of cars where prediction and purchase agree",
        "Of cars predicted NOT due for oil, share that bought oil anyway"],
        "value": [tp / (tp + fp), tp / (tp + fn), (tp + tn) / (tp + fp + fn + tn), fn / (fn + tn)]})

    # 3) km driven since last visit: predicted band vs actual band (clean odometers only)
    ok = df[df["status"] == "ok"]
    km = pd.crosstab(pd.cut(ok["predicted_km_since_last"], KM_BANDS, labels=KM_BAND_LABELS, right=False),
                     pd.cut(ok["actual_km_since_last"], KM_BANDS, labels=KM_BAND_LABELS, right=False),
                     dropna=False).reindex(index=KM_BAND_LABELS, columns=KM_BAND_LABELS, fill_value=0)
    km.index.name = "Predicted km since last visit"
    diag = np.trace(km.values)
    near = sum(km.values[i, j] for i in range(len(km)) for j in range(len(km)) if abs(i - j) <= 1)
    km_metrics = pd.DataFrame({"metric": ["Exact band (diagonal)", "Exact or neighbouring band"],
                               "value": [diag / km.values.sum(), near / km.values.sum()]})
    return {"cm_product": prod, "cm_oil": oil, "cm_oil_metrics": oil_metrics, "cm_km": km, "cm_km_metrics": km_metrics}


# ---------------------------------------------------------------- excel
DETAIL_COLS = {
    "PlateNumber": "Plate", "mobile": "Listed mobile", "Mobile": "Invoice mobile", "match_type": "Match type",
    "campaign_group": "Group", "loyalty_segment": "Loyalty", "price_segment": "Price segment",
    "churn_bucket": "Churn bucket", "Product": "Product", "actual_visit": "Came for", "maker": "Make", "model": "Model", "car_age": "Car age",
    "InvoiceNumber": "First invoice", "InvoiceDate": "Visit date", "visits_in_window": "Visits in window",
    "model_last_visit_dt": "Last visit (model)", "days_since_model_last_visit": "Days since last visit",
    "customer_last_current_mileage": "Last odometer (model)", "customer_avg_mileage_per_day": "Predicted km/day",
    "actual_km_per_day": "Actual km/day", "predicted_at_visit": "Predicted km at visit",
    "actual_mileage": "Actual km at visit", "error_km": "Error km (actual - predicted)",
    "error_pct": "Error %", "error_band": "Error band",
    "predicted_month_end": "Model forecast (month end)", "error_vs_month_end_km": "Error vs month-end km",
    "status": "Status",
}


def write_excel(path: Path, df: pd.DataFrame, tabs: dict, meta: dict) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as xw:
        wb = xw.book
        h = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white", "border": 1,
                           "text_wrap": True, "valign": "vcenter"})
        title = wb.add_format({"bold": True, "font_size": 14})
        bold = wb.add_format({"bold": True})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        km = wb.add_format({"num_format": "#,##0"})
        km1 = wb.add_format({"num_format": "#,##0.0"})
        pct = wb.add_format({"num_format": "0.0%"})

        # README
        ws = wb.add_worksheet("README")
        ws.set_column(0, 0, 34); ws.set_column(1, 1, 110, wrap)
        ws.write(0, 0, f"Mileage prediction vs actual — {meta['target_month']} wave {meta['wave']}", title)
        rows = [
            ("Window", f"{meta['start']} to {meta['end']} (inclusive), M-invoices only (MACDB)"),
            ("List", meta["list"]),
            ("Prediction source", f"ftr_mileage at snapshot {meta['snapshot']} (mileage model run of {meta['run']})"),
            ("Converted car", "An invoice whose PLATE is on the list. 'plate + mobile' = invoice mobile is the listed mobile "
                              "too; 'plate only' = same car, different mobile. Mobile-only matches are excluded."),
            ("First visit only", "Each car is counted once, on its first invoice in the window. Later visits are ignored."),
            ("Actual km at visit", "WorkOrderMileage on that first invoice."),
            ("Predicted km at visit", "Last odometer the model knew + predicted km/day x days from that visit to this visit. "
                                      "The model's own formula, evaluated on the real visit date."),
            ("Model forecast (month end)", "customer_mileage_forecast as the model stored it: projected to the snapshot "
                                           "month-end, so it is ahead of an early-month visit by design."),
            ("Error km", "Actual - predicted. Positive = the car came with MORE km than predicted (drove more / later)."),
            ("Status", "Only 'ok' rows are in the error statistics. Others: no prediction (no odometer history), "
                       "no odometer on the invoice, odometer below the last reading (typo / meter change), or an "
                       f"odometer implying > {MAX_KM_PER_DAY} km/day since the last visit (typo; the model caps at the same value)."),
            ("Coverage", "Every car on the list has a forecast. Cars without one (no odometer history) never reach a list."),
        ]
        for i, (k, v) in enumerate(rows, start=2):
            ws.write(i, 0, k, bold); ws.write(i, 1, v)

        # Summary
        ov = tabs["overall"].reset_index().rename(columns={"index": "metric"})
        ov.to_excel(xw, sheet_name="Summary", index=False, startrow=1)
        ws = xw.sheets["Summary"]
        ws.write(0, 0, "Prediction error on first visit (actual - predicted at visit date)", title)
        for j, c in enumerate(ov.columns):
            ws.write(1, j, c, h)
        ws.set_column(0, 0, 40); ws.set_column(1, len(ov.columns), 18)
        for i, m in enumerate(ov["metric"], start=2):
            f = pct if ("%" in m or "within" in m or "higher" in m) else (km1 if "km/day" in m else km)
            ws.set_row(i, None, f)
            ws.write(i, 0, m, bold)
        r0 = len(ov) + 4
        ws.write(r0 - 1, 0, "Cars by status", bold)
        tabs["status"].to_excel(xw, sheet_name="Summary", index=False, startrow=r0)
        for j, c in enumerate(tabs["status"].columns):
            ws.write(r0, j, c, h)

        # By segment
        bs = tabs["by_segment"]
        bs.to_excel(xw, sheet_name="By segment", index=False)
        ws = xw.sheets["By segment"]
        for j, c in enumerate(bs.columns):
            ws.write(0, j, c, h)
            f = pct if ("%" in c or "within" in c or "higher" in c) else (km1 if "km/day" in c else km)
            ws.set_column(j, j, 22 if j < 2 else 15, None if j < 2 else f)
        ws.set_row(0, 45); ws.freeze_panes(1, 2); ws.autofilter(0, 0, len(bs), len(bs.columns) - 1)

        # Error bands
        b = tabs["bands"]
        b.to_excel(xw, sheet_name="Error bands", index=False)
        ws = xw.sheets["Error bands"]
        for j, c in enumerate(b.columns):
            ws.write(0, j, c, h)
        ws.set_column(0, 0, 30); ws.set_column(1, 1, 10, km); ws.set_column(2, 2, 10, pct); ws.set_column(3, 4, 10, km)
        ws.conditional_format(1, 2, len(b), 2, {"type": "data_bar", "bar_color": "#5B9BD5"})

        # Confusion matrix
        ws = wb.add_worksheet("Confusion matrix")
        xw.sheets["Confusion matrix"] = ws
        ws.set_column(0, 0, 62); ws.set_column(1, 8, 15)
        note = wb.add_format({"italic": True, "font_color": "#595959", "text_wrap": True, "valign": "top"})

        def matrix(r, heading, explain, m, metrics=None):
            ws.write(r, 0, heading, title); ws.merge_range(r + 1, 0, r + 1, 8, explain, note); ws.set_row(r + 1, 45)
            r += 3
            for block, fmt, label in [(m, km, "Cars"), (m.div(m.sum(axis=1).replace(0, np.nan), axis=0), pct, "Row % (each predicted row = 100%)")]:
                ws.write(r, 0, f"{m.index.name or 'Predicted'}  |  {label}", h)
                for j, c in enumerate(block.columns, start=1):
                    ws.write(r, j, f"Actual: {c}" if not str(c).startswith("Actual") else c, h)
                ws.write(r, len(block.columns) + 1, "Total", h)
                for i, (idx, row) in enumerate(block.iterrows(), start=1):
                    ws.write(r + i, 0, idx, bold)
                    for j, v in enumerate(row, start=1):
                        ws.write(r + i, j, 0 if pd.isna(v) else v, fmt)
                    ws.write(r + i, len(row) + 1, m.loc[idx].sum(), km)
                if fmt is pct:
                    ws.conditional_format(r + 1, 1, r + len(block), len(block.columns),
                                          {"type": "2_color_scale", "min_color": "#FFFFFF", "max_color": "#5B9BD5"})
                r += len(block) + 2
            if metrics is not None:
                for i, (mname, val) in enumerate(metrics.itertuples(index=False)):
                    ws.write(r + i, 0, mname, bold); ws.write(r + i, 1, val, pct)
                r += len(metrics) + 1
            return r + 1

        r = matrix(0, "1. Predicted product vs what the car came for",
                   "Rows = what the list said the car was due for. Columns = its first visit: PMS (IsPMS invoice), "
                   "Oil (an oil item on the invoice), or Non Oil only. Converted cars only: cars that did not come "
                   "are not here, so this cannot say how many 'due' cars stayed away.", tabs["cm_product"])
        r = matrix(r, "2. Oil due vs oil bought",
                   "'Due for oil' = Due PMS, Due Oil or Due Oil + Non Oil. Oil is on about 9 in 10 of ALL invoices, "
                   "so high precision is easy; the telling number is the last one: cars the model said were NOT due "
                   "for oil that bought oil anyway.", tabs["cm_oil"], tabs["cm_oil_metrics"])
        matrix(r, "3. Km driven since last visit: predicted band vs actual band",
               "Clean odometer readings only. The diagonal is where the predicted distance band was right; cells to "
               "the left of it = the car drove LESS than predicted, to the right = MORE.", tabs["cm_km"], tabs["cm_km_metrics"])

        # Detail
        d = df[[c for c in DETAIL_COLS if c in df.columns]].rename(columns=DETAIL_COLS)
        d["Error band"] = d["Error band"].astype("string")
        d.to_excel(xw, sheet_name="Detail", index=False)
        ws = xw.sheets["Detail"]
        for j, c in enumerate(d.columns):
            ws.write(0, j, c, h)
            f = pct if c == "Error %" else (km1 if "km/day" in c else (km if ("km" in c.lower() or "odometer" in c.lower()) else None))
            ws.set_column(j, j, 16, f)
        ws.set_row(0, 32); ws.freeze_panes(1, 1); ws.autofilter(0, 0, len(d), len(d.columns) - 1)
        ec = list(d.columns).index("Error km (actual - predicted)")
        ws.conditional_format(1, ec, len(d), ec, {"type": "3_color_scale", "min_color": "#F8696B",
                                                  "mid_type": "num", "mid_value": 0, "mid_color": "#FFFFFF",
                                                  "max_color": "#5A8AC6"})


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, type=Path)
    ap.add_argument("--target-month", required=True)
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--snapshot", help="ftr_mileage _observ_end_dt to compare against (default: month-end of --target-month)")
    a = ap.parse_args()

    snapshot = pd.Timestamp(a.snapshot) if a.snapshot else pd.Timestamp(f"1 {a.target_month}") + pd.offsets.MonthEnd(0)
    lst = load_list(a.list)
    inv = fetch_invoices(a.start, a.end)
    fv = first_visits(lst, inv)
    pred = load_predictions(snapshot)
    df = compare(fv, pred)
    tabs = build_tables(df)

    out_dir = OUTPUTS / pd.Timestamp(f"1 {a.target_month}").strftime("%b-%Y").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"mileage_accuracy_wave_{a.wave}_{a.start}_to_{a.end}.xlsx"
    run = pd.Timestamp(os.path.getmtime(FTR_MILEAGE), unit="s").strftime("%Y-%m-%d")
    write_excel(out, df, tabs, {"target_month": a.target_month, "wave": a.wave, "start": a.start, "end": a.end,
                                "list": str(a.list), "snapshot": snapshot.date(), "run": run})

    print(f"M-invoices in window: {len(inv):,} | plate-matched first visits: {len(df):,} "
          f"{df['match_type'].value_counts().to_dict()}")
    print(df["status"].value_counts().to_string())
    with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
        print(tabs["overall"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
