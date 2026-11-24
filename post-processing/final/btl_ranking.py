"""Oil-visit ranking for the BTL list: who is most likely to come for an oil change this month.

Every active car is placed in a cell of two bands:
  time band  days since its last oil change / the car's usual days between oil changes
             (its own median gap with 2+ gaps, else the median of cars on the same oil type)
  km band    predicted km since its last oil change at the target month end / 5,000 (mineral) or
             10,000 (synthetic)
The cell's score is the share of cars in that cell that came 10th-23rd of the month, measured on
the two latest months whose window is complete in the data. Cars whose cell scores at least
SCORE_MULTIPLIER x the average of all active cars are selected; that is what sets the list size.

Learned from the same history the model uses (data/03_primary/prm_transactions), replayed the same
way as for the target month: data cut on the 18th of the month before, list on the 10th,
customers with a visit between the two removed (the recent-sales cleaning).
Back-test: post-processing/conversions/backtest_ranking.py (Apr-Sep 2026: 13.6% came vs 10.4% for the
oil-due rule at the same list size).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from btl_paths import OUTPUTS, REPO

log = logging.getLogger("btl_ranking")

PRM_TRANSACTIONS = REPO / "data/03_primary/prm_transactions"
CACHE = OUTPUTS / "cache/visits.parquet"
DEFAULT_RATE = 55          # km/day when a car has no usable history (as the model)
MAX_RATE = 700             # km/day above this is an odometer typo (as the model)
JUNK_PLATE_MOBILES = 10    # a plate used by more mobiles than this is a placeholder (0000000 ...)
SCORE_MULTIPLIER = 1.2     # select cells scoring >= this x the average. Nov 2026 replay: 1.1 -> 306K after cleaning (10.6% came), 1.2 -> 242K (11.6%)
BINS = [-np.inf, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, np.inf]
LABELS = ["<50%", "50-80%", "80-100%", "100-120%", "120-150%", "150-200%", "200-300%", ">300%"]
NO_HISTORY = "no oil history"


# ---------------------------------------------------------------- visit history
def build_visits(out: Path = CACHE) -> None:
    import polars as pl
    lf = pl.scan_parquet(str(PRM_TRANSACTIONS / "*.parquet"))
    v = (lf.filter(pl.col("transaction_id").is_not_null() & pl.col("transaction_dt").is_not_null())
         .group_by(["_id", "transaction_id"])
         .agg(dt=pl.col("transaction_dt").min().dt.date(), km=pl.col("current_mileage").max(),
              oil=pl.col("major_category").is_in(["oil", "oil_synthetic"]).any(),
              syn=(pl.col("major_category") == "oil_synthetic").any())
         .collect(engine="streaming"))
    out.parent.mkdir(parents=True, exist_ok=True)
    v.write_parquet(out)


def load_visits(path: Path = CACHE, rebuild: bool | None = None) -> pd.DataFrame:
    """One row per car-customer (_id) per day; rebuilt when prm_transactions is newer than the cache."""
    if rebuild is None:
        newest = max((p.stat().st_mtime for p in PRM_TRANSACTIONS.glob("*.parquet")), default=0)
        rebuild = not path.exists() or path.stat().st_mtime < newest
    if rebuild:
        log.info("building visit history from %s", PRM_TRANSACTIONS)
        build_visits(path)
    v = pd.read_parquet(path, columns=["_id", "dt", "km", "oil", "syn"])
    v["dt"] = pd.to_datetime(v["dt"])
    v.loc[~(v["km"] > 0), "km"] = np.nan
    v = v.groupby(["_id", "dt"], as_index=False, sort=False).agg(
        km=("km", "max"), oil=("oil", "max"), syn=("syn", "max"))
    parts = v["_id"].str.split("__", n=1, expand=True)
    v["plate"], v["mobile"] = parts[0], parts[1]
    n_mob = v.groupby("plate")["mobile"].transform("nunique")
    v = v[n_mob <= JUNK_PLATE_MOBILES].reset_index(drop=True)
    log.info("visit history: %s visits, %s .. %s", f"{len(v):,}", v["dt"].min().date(), v["dt"].max().date())
    return v


# ---------------------------------------------------------------- features
def month_dates(month: str) -> dict:
    """Replay dates for a past month (YYYY-MM): cut 18th of M-1, list 10th, window 10th-23rd."""
    m = pd.Timestamp(f"{month}-01")
    return {"month": month, "cut": m - pd.offsets.MonthBegin(1) + pd.Timedelta(days=17),
            "list": m + pd.Timedelta(days=9), "end": m + pd.Timedelta(days=22),
            "me": m + pd.offsets.MonthEnd(0), "me_prev": m - pd.Timedelta(days=1)}


def features(v: pd.DataFrame, d: dict) -> pd.DataFrame:
    """Per active car at the data cut d['cut']: time / km bands, the model's oil-due flag, removal and outcome."""
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
    cars = cars.join(go.agg(last_oil_dt=("dt", "last"), last_oil_km=("km", "last"), syn=("syn", "last")))
    cars = cars.join(oil.groupby("_id")["gap"].agg(own_days="median", n_gaps="count"))
    has = cars["last_oil_dt"].notna()
    pop = cars.loc[cars["n_gaps"] >= 2].groupby("syn")["own_days"].median()
    cars["usual_oil_gap_days"] = np.where(cars["n_gaps"] >= 2, cars["own_days"], cars["syn"].map(pop))
    cars["days_since_oil"] = (d["list"] - cars["last_oil_dt"]).dt.days
    cars["time_ratio"] = cars["days_since_oil"] / cars["usual_oil_gap_days"]

    interval = np.where(cars["syn"] == True, 10000, 5000)  # noqa: E712
    ks = lambda t: cars["last_km"] + cars["rate"] * (t - cars["last_km_dt"]).dt.days - cars["last_oil_km"]  # noqa: E731
    ks_m, ks_p = ks(d["me"]), ks(d["me_prev"])
    cars["km_ratio"] = ks_m / interval
    f_m, f_p = np.floor(ks_m / interval), np.floor(ks_p / interval)
    cars["current_due"] = has & (f_m >= 1) & (f_m > f_p)

    cars["time_band"] = pd.cut(cars["time_ratio"], BINS, labels=LABELS).astype(object).fillna(NO_HISTORY)
    cars["km_band"] = pd.cut(cars["km_ratio"], BINS, labels=LABELS).astype(object).fillna(NO_HISTORY)

    # recent-sales removal and outcome, by plate or mobile (empty for a future month)
    def keys(lo, hi):
        w = v[(v["dt"] > lo) & (v["dt"] <= hi)]
        return set(w["plate"]), set(w["mobile"])
    rp, rm = keys(d["cut"], d["list"] - pd.Timedelta(days=1))
    cars["removed"] = cars["plate"].isin(rp) | cars["mobile"].isin(rm)
    if "end" in d:
        cp, cm = keys(d["list"] - pd.Timedelta(days=1), d["end"])
        cars["came"] = cars["plate"].isin(cp) | cars["mobile"].isin(cm)
    cars["month"] = d["month"]
    return cars


def cell_scores(train: pd.DataFrame) -> tuple[pd.Series, float]:
    t = train[~train["removed"]]
    return t.groupby(["time_band", "km_band"])["came"].mean().rename("oil_visit_score"), float(t["came"].mean())


# ---------------------------------------------------------------- production
def score_target(target_month: str, as_of, v: pd.DataFrame | None = None, train_months: int = 2) -> pd.DataFrame:
    """Score every active car for the target month. Returns one row per _id with bands, score and selection."""
    v = load_visits() if v is None else v
    data_end = v["dt"].max()
    me = pd.Timestamp(f"1 {target_month}") + pd.offsets.MonthEnd(0)
    target = {"month": me.strftime("%Y-%m"), "cut": data_end, "list": pd.Timestamp(as_of), "me": me,
              "me_prev": me - pd.offsets.MonthEnd(1)}

    months, m = [], data_end.to_period("M")
    while len(months) < train_months:
        if month_dates(str(m))["end"] <= data_end:
            months.append(str(m))
        m -= 1
    train = pd.concat([features(v, month_dates(x)) for x in months])
    scores, base = cell_scores(train)
    log.info("score learned on %s: average %.2f%% came; threshold %.2f%%",
             ", ".join(months), 100 * base, 100 * base * SCORE_MULTIPLIER)

    cars = features(v, target).join(scores, on=["time_band", "km_band"])
    cars["oil_visit_score"] = cars["oil_visit_score"].fillna(base)
    cars["ranked_in"] = cars["oil_visit_score"] >= base * SCORE_MULTIPLIER
    log.info("ranking: %s of %s active cars selected", f"{cars['ranked_in'].sum():,}", f"{len(cars):,}")
    return cars[["oil_visit_score", "ranked_in", "time_band", "km_band", "days_since_oil", "usual_oil_gap_days"]]
