"""Existing cars that came in since the model's data cutoff: were they on our list, and if not, why?

Population: every car (plate) with an M-invoice between the model's data cutoff and --end that the
model already knew with >= --min-visits previous visits. First visit in the window only.

Each car is traced through the same stages the list went through and lands in exactly one bucket:

  1. On the wave list (sent)                         plate is on the wave list
  2. Customer on the wave list with another car      mobile is on the wave list, a different car was listed
  3. On the month list, removed in wave cleaning     recent sale before the send / no consent / NTSC / other lists
  4. Model: Lost/Gone or no visit in 12 months       dropped by the list's "active customers" rule
  5. Model: nothing due this month                   no product assigned (with the reason why, see below)
  6. Due in the model, not on the list               another car of the same mobile was chosen, or mobile changed

For bucket 5 the oil rule is replayed against the real odometer:
  - "passed the oil interval in an earlier month"   the model flags a car only in the month its predicted km
                                                     crosses a 5,000 (mineral) / 10,000 (synthetic) multiple
  - "drove more than predicted"                      predicted below the interval, actual odometer past it
  - "came before the km interval"                    actual odometer still below the interval (time/habit driven)

Usage (BTL env, repo root):
    python post-processing/conversions/visitor_funnel.py \\
        --wave-list "post-processing/final/outputs/nov-2026/nov_2026_wave_1_MASTER_2026-11-10.csv" \\
        --month-list "post-processing/final/outputs/nov-2026/BTL_Targetlist_Nov 2026.csv" \\
        --target-month "Nov 2026" --list-date 2026-11-10 --end 2026-11-23

Output: post-processing/conversions/outputs/<mon-yyyy>/visitor_funnel_<start>_to_<end>.xlsx (git-ignored).
MACDB is only read (SELECT), M-invoices only.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "final"))
sys.path.insert(0, str(HERE))
import btl_targetlist_clean as clean  # noqa: E402
import btl_targetlist_create as create  # noqa: E402
from btl_results_dashboard import load_list, norm_mobile, norm_plate  # noqa: E402
from mileage_accuracy import MAX_KM_PER_DAY, fetch_invoices  # noqa: E402

OUTPUTS = HERE / "outputs"
EXCLUSION_LOG = HERE.parent / "final/outputs/exclusion_log.csv"
JUNK_PLATE_MOBILES = 10      # a "plate" shared by more mobiles than this is a placeholder (0000000 ...)

B_SENT = "1. On the wave list (sent)"
B_OTHER_CAR = "2. Customer on the wave list with another car"
B_CLEANED = "3. On the month list, removed in wave cleaning"
B_INACTIVE = "4. Model: Lost/Gone or no visit in 12 months"
B_NOT_DUE = "5. Model: nothing due this month"
B_DUE_UNLISTED = "6. Due in the model, not on the list"
BUCKETS = [B_SENT, B_OTHER_CAR, B_CLEANED, B_INACTIVE, B_NOT_DUE, B_DUE_UNLISTED]


# ---------------------------------------------------------------- model side
def snapshot_table(folder: str, snapshot: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
    d = ds.dataset(REPO / "data/04_feature" / folder, format="parquet")
    return d.to_table(columns=["_id", *cols], filter=pc.field("_observ_end_dt") == snapshot.date()).to_pandas()


def load_model(snapshot: pd.Timestamp, list_date: date) -> pd.DataFrame:
    fc = create.load_forecast(create.DEFAULT_INPUT)
    fc["customer_last_transaction_dt"] = pd.to_datetime(fc["customer_last_transaction_dt"])
    cutoff = pd.Timestamp(list_date) - pd.Timedelta(days=create.RECENCY_DAYS)
    fc["is_active"] = ~fc["loyalty_segment"].isin(create.EXCLUDED_SEGMENTS) & (fc["customer_last_transaction_dt"] > cutoff)
    prod = create.assign_product(fc.copy())[["_id", "Product", "Sub Product"]]
    m = fc[["_id", "loyalty_segment", "price_segment", "customer_last_transaction_dt", "is_active"]].merge(prod, on="_id", how="left")
    m["Product"] = m["Product"].fillna("")

    seg = snapshot_table("ftr_segment", snapshot, ["total_number_of_visits"])
    mil = snapshot_table("ftr_mileage", snapshot, [
        "customer_last_current_mileage", "customer_avg_mileage_per_day", "customer_mileage_forecast",
        "oil_last_mileage", "oil_synthetic_last_mileage", "customer_mineral_oil", "customer_synthetic_oil",
        "delta_last_mineral_oil_change", "delta_last_synthetic_oil_change",
        "floor_last_delta_last_mineral_oil_change", "floor_delta_last_mineral_oil_change",
        "floor_last_delta_last_synthetic_oil_change", "floor_delta_last_synthetic_oil_change"])
    m = m.merge(seg, on="_id", how="left").merge(mil, on="_id", how="left")

    parts = m["_id"].str.split("__", n=1, expand=True)            # _id = <plate>__<mobile>
    m["plate_key"], m["model_mobile"] = norm_plate(parts[0]), norm_mobile(parts[1])
    m["total_number_of_visits"] = m["total_number_of_visits"].fillna(0)
    g = m.groupby("plate_key")
    m["plate_prev_visits"] = g["total_number_of_visits"].transform("sum")
    m["plate_mobiles"] = g["model_mobile"].transform("nunique")

    # one oil track per car: synthetic if that was the latest oil, else mineral
    syn = (m["customer_synthetic_oil"] == 1).to_numpy()
    m["oil_type"] = np.select([syn, (m["customer_mineral_oil"] == 1).to_numpy()], ["synthetic", "mineral"], "none")
    m["oil_interval"] = np.where(syn, 10000, 5000)
    m["oil_last_km"] = np.where(syn, m["oil_synthetic_last_mileage"], m["oil_last_mileage"])
    m["pred_km_since_oil"] = np.where(syn, m["delta_last_synthetic_oil_change"], m["delta_last_mineral_oil_change"])
    m["floor_prev"] = np.where(syn, m["floor_last_delta_last_synthetic_oil_change"], m["floor_last_delta_last_mineral_oil_change"])
    m["floor_now"] = np.where(syn, m["floor_delta_last_synthetic_oil_change"], m["floor_delta_last_mineral_oil_change"])
    return m


# ---------------------------------------------------------------- wave-cleaning reasons
def cleaning_sets(target_month: str) -> dict[str, tuple[set, set]]:
    """Replay the exclusion sources of the month's last cleaning run (from exclusion_log.csv)."""
    log_df = pd.read_csv(EXCLUSION_LOG, dtype=str)
    log_df = log_df[log_df["target_month"] == target_month]
    last = log_df[log_df["run_at"] == log_df["run_at"].max()]
    engine = clean.get_engine()
    out: dict[str, tuple[set, set]] = {"no consent": (clean.no_consent(engine), set())}
    for r in last.itertuples():
        if str(r.source).startswith("NTSC fleet"):
            since = date.fromisoformat(re.search(r"since (\d{4}-\d{2}-\d{2})", r.source).group(1))
            out["NTSC fleet contract"] = clean.ntsc_fleet(engine, clean.ntsc_contracts(Path(r.path)), since)
        elif isinstance(r.path, str) and r.path:
            spec = r.path if r.sheets in ("(all)", "", None) or pd.isna(r.sheets) else f"{r.path}#{r.sheets}"
            out[f"already on another list ({Path(r.path).name})"] = clean.read_list(spec)
    return out


# ---------------------------------------------------------------- classification
def classify(v: pd.DataFrame, wave: pd.DataFrame, month: pd.DataFrame, clean_sets: dict,
             pre_send: pd.DataFrame, since_recent: pd.Timestamp) -> pd.DataFrame:
    wave_pl, wave_mob = set(wave["plate_key"].dropna()), set(wave["mobile_key"].dropna())
    month_pl, month_mob = set(month["plate_key"].dropna()), set(month["mobile_key"].dropna())
    # the customer's keys: invoice mobile, the model's mobile for this car, and the plate
    on = lambda mobs, pls: (v["mobile_key"].isin(mobs) | v["model_mobile"].isin(mobs) | v["plate_key"].isin(pls))  # noqa: E731

    v["is_active"] = v["is_active"].fillna(False).astype(bool)
    v["exact_id"] = v["exact_id"].fillna(False).astype(bool)
    in_wave_plate = v["plate_key"].isin(wave_pl)
    in_wave_mob = v["mobile_key"].isin(wave_mob) | v["model_mobile"].isin(wave_mob)
    in_month = on(month_mob, month_pl)
    v["bucket"] = np.select(
        [in_wave_plate, in_wave_mob, in_month, ~v["is_active"], v["Product"] == ""],
        [B_SENT, B_OTHER_CAR, B_CLEANED, B_INACTIVE, B_NOT_DUE], B_DUE_UNLISTED)

    # reason detail -------------------------------------------------------
    reason = pd.Series("", index=v.index, dtype="object")

    grp = wave.drop_duplicates("plate_key").set_index("plate_key")["campaign_group"]
    reason[v["bucket"] == B_SENT] = v["plate_key"].map(grp).fillna("")

    recent = pre_send[pre_send["InvoiceDate"] >= since_recent]
    recent_hit = on(set(recent["mobile_key"].dropna()), set(recent["plate_key"].dropna()))
    conds = [recent_hit] + [on(mobs, pls) for mobs, pls in clean_sets.values()]
    names = [f"came {since_recent:%d %b} to the day before the send (removed as a recent sale)", *clean_sets.keys()]
    cleaned = v["bucket"] == B_CLEANED
    reason[cleaned] = pd.Series(np.select(conds, names, "other / not reproduced"), index=v.index)[cleaned]

    reason[v["bucket"] == B_INACTIVE] = np.where(
        v["loyalty_segment"].isin(create.EXCLUDED_SEGMENTS), "segment " + v["loyalty_segment"].astype(str),
        "last visit more than 12 months before the list date")[v["bucket"] == B_INACTIVE]

    v["actual_km_since_oil"] = v["actual_mileage"] - v["oil_last_km"]
    days = (v["InvoiceDate"].dt.normalize() - v["customer_last_transaction_dt"].dt.normalize()).dt.days
    v["days_since_last_visit"] = days
    bad_odo = (v["actual_mileage"].isna() | (v["actual_mileage"] <= 0)
               | (v["actual_mileage"] < v["customer_last_current_mileage"])
               | ((v["actual_mileage"] - v["customer_last_current_mileage"]) / days.clip(lower=1) > MAX_KM_PER_DAY))
    not_due = np.select(
        [v["actual_visit"] == "Non Oil only",
         (v["oil_type"] == "none") | v["oil_last_km"].isna(),
         (v["floor_now"] >= 1) & (v["floor_now"] == v["floor_prev"]),
         bad_odo,
         v["actual_km_since_oil"] >= v["oil_interval"],
         v["actual_km_since_oil"] < v["oil_interval"]],
        ["came for non-oil only (no non-oil product was due)",
         "no oil-change history on this car in the model",
         "passed the oil interval in an earlier month (flagged then, not re-flagged)",
         "odometer on the invoice unusable (typo)",
         "drove more than predicted (really past the interval)",
         "came before the km interval (time / habit driven)"],
        "other")
    reason[v["bucket"] == B_NOT_DUE] = pd.Series(not_due, index=v.index)[v["bucket"] == B_NOT_DUE]

    reason[v["bucket"] == B_DUE_UNLISTED] = np.where(
        v["exact_id"], "due under this mobile, not listed (other car chosen / list-date difference)",
        "car is known under a different mobile; that mobile was not listed")[v["bucket"] == B_DUE_UNLISTED]
    v["reason"] = reason
    return v


# ---------------------------------------------------------------- excel
def pivot(df: pd.DataFrame, rows: list[str], cols: str) -> pd.DataFrame:
    t = pd.crosstab([df[r] for r in rows], df[cols], margins=True, margins_name="Total")
    return t.reset_index()


def write_excel(path: Path, v: pd.DataFrame, everyone: pd.DataFrame, meta: dict) -> dict:
    periods = [meta["p_before"], meta["p_after"]]
    funnel = pivot(v, ["bucket"], "period").rename(columns={"bucket": "Where the car ended up"})
    funnel["Share"] = funnel["Total"] / len(v)
    reasons = pivot(v, ["bucket", "reason"], "period")
    reasons["Share of bucket"] = reasons["Total"] / reasons.groupby("bucket")["Total"].transform("sum")
    reasons = reasons.rename(columns={"bucket": "Where the car ended up", "reason": "Detail / reason"})
    nd = v[v["bucket"] == B_NOT_DUE]
    nd_tab = pivot(nd, ["reason"], "actual_visit").rename(columns={"reason": "Why nothing was due"})
    nd_tab["Share"] = nd_tab["Total"] / len(nd)
    nd_stats = nd.groupby("reason").agg(
        cars=("plate_key", "size"),
        median_days_since_last_visit=("days_since_last_visit", "median"),
        median_predicted_km_since_oil=("pred_km_since_oil", "median"),
        median_actual_km_since_oil=("actual_km_since_oil", "median"),
        median_predicted_km_per_day=("customer_avg_mileage_per_day", "median"),
    ).reset_index().rename(columns={"reason": "Why nothing was due"})

    detail_cols = {
        "PlateNumber": "Plate", "Mobile": "Invoice mobile", "model_mobile": "Mobile in model", "period": "Period",
        "bucket": "Where the car ended up", "reason": "Detail / reason", "InvoiceNumber": "First invoice",
        "InvoiceDate": "Visit date", "actual_visit": "Came for", "plate_prev_visits": "Previous visits (car)",
        "loyalty_segment": "Loyalty", "Product": "Model product", "customer_last_transaction_dt": "Last visit (model)",
        "days_since_last_visit": "Days since last visit", "oil_type": "Oil type", "oil_interval": "Oil interval km",
        "pred_km_since_oil": "Predicted km since oil (month end)", "actual_km_since_oil": "Actual km since oil",
        "customer_avg_mileage_per_day": "Predicted km/day", "customer_last_current_mileage": "Last odometer (model)",
        "actual_mileage": "Actual odometer", "revenue": "Invoice value (before VAT)",
    }
    d = v[list(detail_cols)].rename(columns=detail_cols)

    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        wb = xw.book
        h = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white", "border": 1, "text_wrap": True, "valign": "vcenter"})
        title = wb.add_format({"bold": True, "font_size": 14})
        bold = wb.add_format({"bold": True})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        num = wb.add_format({"num_format": "#,##0"})
        pct = wb.add_format({"num_format": "0.0%"})

        ws = wb.add_worksheet("README")
        ws.set_column(0, 0, 34); ws.set_column(1, 1, 115, wrap)
        ws.write(0, 0, f"Returning cars since the model run: on our list or not, and why — {meta['target_month']}", title)
        rows = [
            ("Question", f"Of the existing cars (>= {meta['min_visits']} previous visits) that came in after the model's data ended, "
                         "how many did we have on the list, and what kept the others off it?"),
            ("Window", f"{meta['start']} to {meta['end']} (inclusive). The model's data ends {meta['cutoff']}. M-invoices only (MACDB)."),
            ("Unit", "One row per car (plate): its FIRST visit in the window. Previous visits = visits the model knew for that plate (all mobiles)."),
            ("Periods", f"'{periods[0]}' = the car came before the list was sent on {meta['list_date']}; '{periods[1]}' = it came after."),
            (B_SENT, "The plate is on the wave list (TEST or CONTROL in the detail column)."),
            (B_OTHER_CAR, "The customer's mobile is on the wave list, but for a different car (the list keeps one car per mobile)."),
            (B_CLEANED, "Was on the month list and removed in wave cleaning. The reason replays the cleaning sources: a sale shortly "
                        "before the send, no consent, NTSC fleet, or already on another marketing list."),
            (B_INACTIVE, "The model had the car as Lost/Gone, or its last visit was more than 12 months before the list date."),
            (B_NOT_DUE, "The model assigned no product for this month. The 'Nothing due' sheet replays the oil rule against the real odometer."),
            (B_DUE_UNLISTED, "The model had a product due, but the car is not on the month list: another car of the same mobile was "
                             "chosen, or the car came in under a mobile the model did not know for it."),
            ("Oil rule", "The model flags oil only in the month the predicted km since the last oil change crosses a 5,000 km (mineral) or "
                         "10,000 km (synthetic) multiple. A car that crossed in an earlier month and did not come is not flagged again "
                         "until the next multiple."),
            ("Left out", f"{meta['junk']:,} cars on placeholder plates (shared by more than {JUNK_PLATE_MOBILES} mobiles)."),
        ]
        for i, (k, val) in enumerate(rows, start=2):
            ws.write(i, 0, k, bold); ws.write(i, 1, val)

        def sheet(name, df_, heading, widths, pct_cols=()):
            df_.to_excel(xw, sheet_name=name, index=False, startrow=2)
            ws_ = xw.sheets[name]
            ws_.write(0, 0, heading, title)
            for j, c in enumerate(df_.columns):
                ws_.write(2, j, c, h)
                ws_.set_column(j, j, widths[j] if j < len(widths) else 14, pct if c in pct_cols else num)
            ws_.set_row(2, 32); ws_.freeze_panes(3, 1)
            return ws_

        sheet("Everyone who came", everyone, "All cars with an M-invoice in the window, by what the model knew", [44, 16, 16, 12, 12], ("Share",))
        sheet("Funnel", funnel, f"Existing cars (>= {meta['min_visits']} previous visits): where each one ended up", [50, 22, 22, 12, 10], ("Share",))
        sheet("Reasons", reasons, "Same cars, with the reason inside each bucket", [50, 72, 22, 22, 12, 14], ("Share of bucket",))
        ws = sheet("Nothing due", nd_tab, "Bucket 5: cars the model said had nothing due, by why and by what they came for", [72, 14, 14, 14, 12, 10], ("Share",))
        r0 = len(nd_tab) + 5
        ws.write(r0 - 1, 0, "Medians per reason", bold)
        nd_stats.to_excel(xw, sheet_name="Nothing due", index=False, startrow=r0)
        for j, c in enumerate(nd_stats.columns):
            ws.write(r0, j, c.replace("_", " "), h)
        ws.set_row(r0, 45)

        d.to_excel(xw, sheet_name="Detail", index=False)
        ws = xw.sheets["Detail"]
        for j, c in enumerate(d.columns):
            ws.write(0, j, c, h)
            ws.set_column(j, j, 44 if c in ("Where the car ended up", "Detail / reason") else 16,
                          num if any(k in c.lower() for k in ("km", "odometer", "value", "visits")) else None)
        ws.set_row(0, 32); ws.freeze_panes(1, 1); ws.autofilter(0, 0, len(d), len(d.columns) - 1)
    return {"funnel": funnel, "reasons": reasons, "nd": nd_tab, "nd_stats": nd_stats}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wave-list", required=True, type=Path)
    ap.add_argument("--month-list", required=True, type=Path)
    ap.add_argument("--target-month", required=True)
    ap.add_argument("--list-date", required=True, help="day the wave list was built / sent, e.g. 2026-11-10")
    ap.add_argument("--end", required=True, help="last day of visits to include")
    ap.add_argument("--start", help="first day of visits (default: day after the model's last transaction)")
    ap.add_argument("--recent-since", help="start of the cleaning's recent-sales window (default: from exclusion_log)")
    ap.add_argument("--min-visits", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    snapshot = pd.Timestamp(f"1 {a.target_month}") + pd.offsets.MonthEnd(0)
    list_date = pd.Timestamp(a.list_date)
    model = load_model(snapshot, list_date.date())
    cutoff = model["customer_last_transaction_dt"].max().normalize()
    start = pd.Timestamp(a.start) if a.start else cutoff + pd.Timedelta(days=1)
    print(f"model data ends {cutoff.date()} | visits {start.date()} .. {a.end}")

    inv = fetch_invoices(start.strftime("%Y-%m-%d"), a.end)
    first = (inv.dropna(subset=["plate_key"]).sort_values(["plate_key", "InvoiceDate", "InvoiceNumber"])
             .drop_duplicates("plate_key").reset_index(drop=True))

    # model row for the car: the exact plate+mobile _id if it exists, else the plate's most recent _id
    cand = first[["plate_key", "mobile_key"]].merge(model, on="plate_key", how="inner")
    cand["exact_id"] = cand["mobile_key"] == cand["model_mobile"]
    cand = (cand.sort_values(["plate_key", "exact_id", "customer_last_transaction_dt"], ascending=[True, False, False])
            .drop_duplicates("plate_key").drop(columns="mobile_key"))
    v = first.merge(cand, on="plate_key", how="left")
    p_before, p_after = f"Came before the send (to {list_date - pd.Timedelta(days=1):%d %b})", f"Came after the send ({list_date:%d %b}+)"
    v["period"] = np.where(v["InvoiceDate"] < list_date, p_before, p_after)

    known = np.select(
        [v["_id"].isna(), v["plate_mobiles"] > JUNK_PLATE_MOBILES, v["plate_prev_visits"] == 0,
         v["plate_prev_visits"] < a.min_visits],
        ["New to the model (plate never seen)", "Placeholder plate (left out)",
         "Known plate, no previous visit recorded", f"Existing, fewer than {a.min_visits} previous visits"],
        f"Existing, {a.min_visits}+ previous visits (analysed)")
    everyone = pd.crosstab(known, v["period"], margins=True, margins_name="Total").reset_index().rename(columns={"row_0": "Car"})
    everyone["Share"] = everyone["Total"] / len(v)

    junk = int((known == "Placeholder plate (left out)").sum())
    v = v[known == f"Existing, {a.min_visits}+ previous visits (analysed)"].copy()

    wave, month = load_list(a.wave_list), load_list(a.month_list)
    if a.recent_since:
        since_recent = pd.Timestamp(a.recent_since)
    else:
        src = pd.read_csv(EXCLUSION_LOG, dtype=str)
        src = src[(src["target_month"] == a.target_month) & src["source"].str.startswith("recent sales")]["source"].iloc[-1]
        since_recent = pd.Timestamp(re.search(r"since (\d{4}-\d{2}-\d{2})", src).group(1))
    v = classify(v, wave, month, cleaning_sets(a.target_month), inv[inv["InvoiceDate"] < list_date], since_recent)

    out_dir = OUTPUTS / pd.Timestamp(f"1 {a.target_month}").strftime("%b-%Y").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"visitor_funnel_{start.date()}_to_{a.end}.xlsx"
    tabs = write_excel(out, v, everyone, {
        "target_month": a.target_month, "min_visits": a.min_visits, "start": start.date(), "end": a.end,
        "cutoff": cutoff.date(), "list_date": list_date.date(), "junk": junk, "p_before": p_before, "p_after": p_after})

    with pd.option_context("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 100):
        print(everyone.to_string(index=False)); print()
        print(tabs["funnel"].round(3).to_string(index=False)); print()
        print(tabs["reasons"].round(3).to_string(index=False)); print()
        print(tabs["nd"].round(3).to_string(index=False)); print()
        print(tabs["nd_stats"].round(0).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
