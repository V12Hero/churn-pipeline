"""
Back-test: today's oil-due rule vs ranking customers by time x km since their last oil change.

Replays past months from the visit history the model is built on (outputs/cache/visits.parquet, see
improvement_analysis.py), no pipeline run needed. For each test month M:

  data cut   18th of M-1   (as the real run: data ends ~3 weeks before the send)
  list date  10th of M
  window     10th-23rd of M; "came" = any visit by the car's plate or the customer's mobile
  removed    customers with any visit between the data cut and the list date (the recent-sales cleaning)
  active     last visit within 365 days of the list date

  current rule  the model's oil flag: predicted km since last oil crosses a 5,000 / 10,000 km multiple in month M
  ranking       every active car is scored by the hit rate of its (time band x km band) cell, learned on the two
                previous months; one row per mobile (best car); the top N are taken, N = size of the current list

Simplified on purpose (both methods equally): oil only (no non-oil products), no Lost/Gone or churn filter,
no consent / NTSC / other-list exclusions. Results print to the terminal.

    python post-processing/conversions/backtest_ranking.py --months 2026-05 2026-06 2026-07 2026-08 2026-09
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from improvement_analysis import CACHE, MAX_RATE, build_cache, load_visits  # noqa: E402

DEFAULT_RATE = 55
BINS = [-np.inf, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, np.inf]
LABELS = ["<50%", "50-80%", "80-100%", "100-120%", "120-150%", "150-200%", "200-300%", ">300%"]


def dates(month: str) -> dict:
    m = pd.Timestamp(f"{month}-01")
    prev = m - pd.offsets.MonthBegin(1)
    return {"month": month, "cut": prev + pd.Timedelta(days=17), "list": m + pd.Timedelta(days=9),
            "end": m + pd.Timedelta(days=22), "me": m + pd.offsets.MonthEnd(0), "me_prev": m - pd.Timedelta(days=1)}


def features(v: pd.DataFrame, d: dict) -> pd.DataFrame:
    h = v[v["dt"] <= d["cut"]]
    cars = h.groupby("_id", sort=False).agg(plate=("plate", "first"), mobile=("mobile", "first"), last_dt=("dt", "max"))
    cars = cars[cars["last_dt"] >= d["list"] - pd.Timedelta(days=365)]
    h = h[h["_id"].isin(cars.index)]

    # km/day: mean of the last 12 valid odometer intervals, default 55 (as the model)
    o = h[h["km"].notna()].sort_values(["_id", "dt"])
    g = o.groupby("_id", sort=False)
    days = (o["dt"] - g["dt"].shift()).dt.days
    rate = (o["km"] - g["km"].shift()) / days
    o = o.assign(rate=rate.where((days >= 1) & (rate >= 0) & (rate <= MAX_RATE)))
    last = g.agg(last_km=("km", "last"), last_km_dt=("dt", "last"))
    r = o.dropna(subset=["rate"]).groupby("_id")["rate"].apply(lambda s: s.iloc[-12:].mean())
    cars = cars.join(last).join(r.rename("rate"))
    cars["rate"] = cars["rate"].fillna(DEFAULT_RATE)

    # oil history
    oil = h[h["oil"]].sort_values(["_id", "dt"])
    go = oil.groupby("_id", sort=False)
    gap = (oil["dt"] - go["dt"].shift()).dt.days
    oil = oil.assign(gap=gap.where(gap >= 14))
    oo = go.agg(last_oil_dt=("dt", "last"), last_oil_km=("km", "last"), syn=("syn", "last"))
    oo = oo.join(oil.groupby("_id")["gap"].agg(own_days="median", n_gaps="count"))
    cars = cars.join(oo)
    has = cars["last_oil_dt"].notna()
    pop = cars.loc[cars["n_gaps"] >= 2].groupby("syn")["own_days"].median()
    usual = np.where(cars["n_gaps"] >= 2, cars["own_days"], cars["syn"].map(pop))
    cars["time_ratio"] = (d["list"] - cars["last_oil_dt"]).dt.days / usual

    interval = np.where(cars["syn"] == True, 10000, 5000)  # noqa: E712
    ks = lambda t: cars["last_km"] + cars["rate"] * (t - cars["last_km_dt"]).dt.days - cars["last_oil_km"]  # noqa: E731
    ks_m, ks_p = ks(d["me"]), ks(d["me_prev"])
    cars["km_ratio"] = ks_m / interval
    f_m, f_p = np.floor(ks_m / interval), np.floor(ks_p / interval)
    cars["current_due"] = has & (f_m >= 1) & (f_m > f_p)

    cars["time_band"] = pd.cut(cars["time_ratio"], BINS, labels=LABELS).astype(object).fillna("no oil history")
    cars["km_band"] = pd.cut(cars["km_ratio"], BINS, labels=LABELS).astype(object).fillna("no oil history")

    # cleaning and outcome, by plate or mobile
    def keys(lo, hi):
        w = v[(v["dt"] > lo) & (v["dt"] <= hi)]
        return set(w["plate"]), set(w["mobile"])
    rp, rm = keys(d["cut"], d["list"] - pd.Timedelta(days=1))
    cars = cars[~(cars["plate"].isin(rp) | cars["mobile"].isin(rm))].copy()
    cp, cm = keys(d["list"] - pd.Timedelta(days=1), d["end"])
    cars["came"] = cars["plate"].isin(cp) | cars["mobile"].isin(cm)
    cars["month"] = d["month"]
    return cars


def per_mobile(df: pd.DataFrame, flag: pd.Series | None = None, score: pd.Series | None = None) -> pd.DataFrame:
    x = df.assign(score=score if score is not None else 0.0)
    if flag is not None:
        x = x[flag]
    came = x.groupby("mobile")["came"].any()
    best = x.groupby("mobile")["score"].max()
    return pd.DataFrame({"came": came, "score": best})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs="+", required=True, help="test months, YYYY-MM (window must end before the data does)")
    a = ap.parse_args()
    if not CACHE.exists():
        build_cache()
    v = load_visits()
    print(f"history to {v['dt'].max().date()}")

    first = pd.Timestamp(f"{min(a.months)}-01")
    need = sorted({*(f"{(first - pd.DateOffset(months=k)):%Y-%m}" for k in (1, 2)), *a.months})
    feats = {}
    for m in need:
        feats[m] = features(v, dates(m))
        print(f"  built {m}: {len(feats[m]):,} active cars")

    rows = []
    for m in a.months:
        t = pd.Timestamp(f"{m}-01")
        train = pd.concat([feats[f"{(t - pd.DateOffset(months=k)):%Y-%m}"] for k in (1, 2)])
        cell = train.groupby(["time_band", "km_band"])["came"].mean().rename("cell")
        test = feats[m].join(cell, on=["time_band", "km_band"])
        test["cell"] = test["cell"].fillna(train["came"].mean())

        cur = per_mobile(test, flag=test["current_due"])
        allm = per_mobile(test, score=test["cell"]).sort_values("score", ascending=False)
        n = len(cur)
        row = {"Month": m, "Active customers": len(allm), "Came (all active)": allm["came"].mean(),
               "Current list": n, "Current came": int(cur["came"].sum()), "Current %": cur["came"].mean()}
        for k, lab in [(1.0, "Ranked, same size"), (0.8, "Ranked, 80% size"), (1.2, "Ranked, 120% size")]:
            top = allm.head(int(n * k))
            row[f"{lab} came"] = int(top["came"].sum())
            row[f"{lab} %"] = top["came"].mean()
        same = allm.head(n)
        row["Overlap with current list"] = same.index.isin(cur.index).mean()
        rows.append(row)

    r = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(r.round(4).to_string(index=False))
        print("\nAll months:")
        tot = {c: r[c].sum() for c in r.columns if "came" in c and "%" not in c or c == "Current list"}
        print(f"  current list {tot['Current list']:,} msgs -> {tot['Current came']:,} came ({tot['Current came'] / tot['Current list']:.2%})")
        print(f"  ranked same  {tot['Current list']:,} msgs -> {tot['Ranked, same size came']:,} came "
              f"({tot['Ranked, same size came'] / tot['Current list']:.2%})")
        print(f"  ranked 80%   {int(0.8 * tot['Current list']):,} msgs -> {tot['Ranked, 80% size came']:,} came")


if __name__ == "__main__":
    main()
