"""
Evidence for four possible model changes, from the visit history the model was built on
(data/03_primary/prm_transactions) and the visits that came after it (MACDB, read only, M-invoices).

  1. Default km/day    is km/day tied to make / model?  what beats the flat 55 km/day for a car with no history?
  2. Long-absent cars  is the over-prediction for long gaps typical, or a few extreme cars?
  4. Plate history     when a plate shows up under a new mobile, does the old history still describe the car?
  5. Time-based rule   do customers change oil on km or on time?  what would a time rule add to the list?

    python post-processing/conversions/improvement_analysis.py \
        --month-list "post-processing/final/outputs/nov-2026/BTL_Targetlist_Nov 2026.csv" \
        --target-month "Nov 2026" --list-date 2026-11-10 --end 2026-11-23

Output: post-processing/conversions/outputs/<mon-yyyy>/improvement_analysis_<target month>.xlsx  (git-ignored)
The visit cache (outputs/cache/visits.parquet) is rebuilt with --rebuild-cache after a new pipeline run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import visitor_funnel as vf  # noqa: E402

CACHE = HERE / "outputs/cache/visits.parquet"
DEFAULT_RATE = 55
MAX_RATE = 700
JUNK_PLATE_MOBILES = 10
SPLIT = pd.Timestamp("2026-01-01")      # priors are learned before this date and tested after it
PCTS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


# ---------------------------------------------------------------- data
def build_cache() -> None:
    import polars as pl
    lf = pl.scan_parquet(str(REPO / "data/03_primary/prm_transactions/*.parquet"))
    v = (lf.filter(pl.col("transaction_id").is_not_null() & pl.col("transaction_dt").is_not_null())
         .group_by(["_id", "transaction_id"])
         .agg(dt=pl.col("transaction_dt").min().dt.date(), km=pl.col("current_mileage").max(),
              oil=pl.col("major_category").is_in(["oil", "oil_synthetic"]).any(),
              syn=(pl.col("major_category") == "oil_synthetic").any(),
              fleet=pl.col("is_fleet").max(), revenue=pl.col("sales_amount_net").sum())
         .collect(engine="streaming"))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    v.write_parquet(CACHE)


def load_visits() -> pd.DataFrame:
    """One row per car-customer (_id) per day, placeholder plates removed."""
    v = pd.read_parquet(CACHE, columns=["_id", "dt", "km", "oil", "syn", "fleet"])
    v["dt"] = pd.to_datetime(v["dt"])
    v.loc[~(v["km"] > 0), "km"] = np.nan
    v = v.groupby(["_id", "dt"], as_index=False, sort=False).agg(
        km=("km", "max"), oil=("oil", "max"), syn=("syn", "max"), fleet=("fleet", "max"))
    parts = v["_id"].str.split("__", n=1, expand=True)
    v["plate"], v["mobile"] = parts[0], parts[1]
    n_mob = v.groupby("plate")["mobile"].transform("nunique")
    v = v[n_mob <= JUNK_PLATE_MOBILES].copy()
    v["plate_mobiles"] = n_mob[n_mob <= JUNK_PLATE_MOBILES]
    return v.sort_values(["plate", "dt", "_id"]).reset_index(drop=True)


def odo_intervals(v: pd.DataFrame) -> pd.DataFrame:
    """Consecutive odometer readings of the same _id."""
    o = v[v["km"].notna()].sort_values(["_id", "dt"]).copy()
    g = o.groupby("_id", sort=False)
    o["prev_dt"], o["prev_km"] = g["dt"].shift(), g["km"].shift()
    o["seq"] = g.cumcount()
    o["days"] = (o["dt"] - o["prev_dt"]).dt.days
    o["rate"] = (o["km"] - o["prev_km"]) / o["days"]
    o["ok"] = (o["days"] >= 30) & (o["rate"] >= 0) & (o["rate"] <= MAX_RATE)
    return o


def car_rates(o: pd.DataFrame) -> pd.DataFrame:
    """Whole-history km/day per _id: (last odometer - first odometer) / days, at least 60 days apart."""
    c = o.groupby("_id", sort=False).agg(first_dt=("dt", "first"), last_dt=("dt", "last"), first_km=("km", "first"),
                                         last_km=("km", "last"), odo_visits=("km", "size"))
    c["span"] = (c["last_dt"] - c["first_dt"]).dt.days
    c["rate"] = (c["last_km"] - c["first_km"]) / c["span"]
    c.loc[~((c["span"] >= 60) & (c["rate"] > 0) & (c["rate"] <= MAX_RATE)), "rate"] = np.nan
    return c


def vehicles(snapshot: pd.Timestamp) -> pd.DataFrame:
    a = vf.snapshot_table("ftr_vehicle", snapshot, ["maker", "model", "model_year", "is_truck"])
    a["model_year"] = pd.to_numeric(a["model_year"], errors="coerce")
    a.loc[~a["model_year"].between(1980, snapshot.year + 1), "model_year"] = np.nan
    for c in ("maker", "model"):
        a[c] = a[c].fillna("").astype(str).str.strip().str.upper().replace({"": "UNKNOWN", "NONE": "UNKNOWN", "NAN": "UNKNOWN"})
    a["make_model"] = a["maker"] + " " + a["model"]
    return a


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, [-np.inf, 2, 5, 9, 14, np.inf], labels=["0-2 yrs", "3-5 yrs", "6-9 yrs", "10-14 yrs", "15+ yrs"]).astype(object).fillna("unknown")


def pct_table(s: pd.Series) -> dict:
    q = s.quantile(PCTS)
    return {f"P{int(p * 100)}": q[p] for p in PCTS}


def score(pred: pd.Series, actual: pd.Series, days: pd.Series) -> dict:
    e, ekm = pred - actual, (pred - actual) * days
    return {"Median abs error km/day": e.abs().median(), "Median error km/day (+ = too high)": e.median(),
            "Median abs error km at the visit": ekm.abs().median(), "% within 2,500 km at the visit": (ekm.abs() <= 2500).mean()}


# ---------------------------------------------------------------- 1. make / model and the default rate
def section_make_model(o: pd.DataFrame, cars: pd.DataFrame, veh: pd.DataFrame) -> dict:
    c = cars.dropna(subset=["rate"]).reset_index().merge(veh, on="_id", how="left")
    c[["maker", "model", "make_model"]] = c[["maker", "model", "make_model"]].fillna("UNKNOWN")
    c["age"] = c["last_dt"].dt.year - c["model_year"]
    c["age_grp"] = age_group(c["age"])
    c["log_rate"] = np.log(c["rate"].clip(lower=1))

    # how much of the car-to-car difference each attribute explains, out of sample (fit on half, score on the other half)
    rng = np.random.default_rng(7)
    half = rng.random(len(c)) < 0.5
    tr, te = c[half], c[~half]
    tot = ((te["log_rate"] - tr["log_rate"].mean()) ** 2).sum()

    def r2(keys):
        gm = tr.groupby(keys)["log_rate"].agg(["mean", "size"]).reset_index()
        gm = gm[gm["size"] >= 30]
        p = te[keys].merge(gm, on=keys, how="left")["mean"].fillna(tr["log_rate"].mean()).to_numpy()
        return 1 - ((te["log_rate"].to_numpy() - p) ** 2).sum() / tot, len(gm)

    rows = []
    for label, keys in [("Make", ["maker"]), ("Make + model", ["make_model"]), ("Car age group", ["age_grp"]),
                        ("Make + model + car age group", ["make_model", "age_grp"]),
                        ("Truck flag", ["is_truck"])]:
        val, n = r2(keys)
        rows.append({"Grouping": label, "Groups": n, "% of car-to-car difference explained": max(val, 0)})
    # benchmark: the same car's own earlier history
    multi = o[o["ok"]].copy()
    multi["n"] = multi.groupby("_id")["rate"].transform("size")
    multi = multi[multi["n"] >= 2]
    last = multi.groupby("_id").tail(1).set_index("_id")["rate"]
    earlier = multi[~multi.index.isin(multi.groupby("_id").tail(1).index)].groupby("_id")["rate"].mean()
    b = pd.concat([np.log(last.clip(lower=1)).rename("y"), np.log(earlier.clip(lower=1)).rename("x")], axis=1).dropna()
    rows.append({"Grouping": "Benchmark: the car's own earlier km/day", "Groups": len(b),
                 "% of car-to-car difference explained": np.corrcoef(b["x"], b["y"])[0, 1] ** 2})
    explained = pd.DataFrame(rows)

    def spread(keys, min_n, top=None):
        t = c.groupby(keys)["rate"].agg(Cars="size", P25=lambda s: s.quantile(.25), Median="median",
                                        P75=lambda s: s.quantile(.75)).reset_index()
        t = t[t["Cars"] >= min_n].sort_values("Cars", ascending=False)
        return t.head(top) if top else t

    by_make = spread(["maker"], 2000, 25).rename(columns={"maker": "Make"})
    mm = spread(["maker", "model"], 2000).rename(columns={"maker": "Make", "model": "Model"})
    by_model = mm.head(30)
    extremes = pd.concat([mm.nlargest(10, "Median"), mm.nsmallest(10, "Median")])
    by_age = spread(["age_grp"], 1).rename(columns={"age_grp": "Car age"})
    overall = pd.DataFrame([{"Cars": len(c), "P25": c["rate"].quantile(.25), "Median": c["rate"].median(),
                             "P75": c["rate"].quantile(.75), "Mean": c["rate"].mean()}])

    # test: the first interval of a car (visit 1 -> visit 2) is where the model has nothing and uses 55
    fi = o[(o["seq"] == 1) & o["ok"]].merge(veh, on="_id", how="left")
    fi[["maker", "make_model"]] = fi[["maker", "make_model"]].fillna("UNKNOWN")
    fi["age_grp"] = age_group(fi["prev_dt"].dt.year - fi["model_year"])
    age_days = (fi["prev_dt"] - pd.to_datetime(fi["model_year"].fillna(1900).astype(int).astype(str) + "-01-01")).dt.days.clip(lower=365)
    fi["lifetime"] = np.where(fi["model_year"].notna(), (fi["prev_km"] / age_days).clip(5, 300), np.nan)
    train, test = fi[fi["dt"] < SPLIT], fi[fi["dt"] >= SPLIT].copy()
    g_med = train["rate"].median()

    def prior(keys, min_n):
        t = train.groupby(keys)["rate"].agg(["median", "size"]).reset_index()
        return test[keys].merge(t[t["size"] >= min_n], on=keys, how="left")["median"].to_numpy()

    test["p_make"] = pd.Series(prior(["maker"], 200), index=test.index).fillna(g_med)
    test["p_mm"] = pd.Series(prior(["make_model"], 100), index=test.index).fillna(test["p_make"])
    test["p_mm_age"] = pd.Series(prior(["make_model", "age_grp"], 100), index=test.index).fillna(test["p_mm"])
    test["p_age"] = pd.Series(prior(["age_grp"], 100), index=test.index).fillna(g_med)
    test["p_life"] = test["lifetime"].fillna(test["p_mm_age"])
    # lifetime average scaled so its median matches recent driving of cars of that age
    k = (train["rate"] / train["lifetime"]).groupby(train["age_grp"]).median()
    test["p_life_adj"] = (test["lifetime"] * test["age_grp"].map(k)).fillna(test["p_mm_age"])
    test["p_blend"] = (test["p_life_adj"] + test["p_mm_age"]) / 2
    cands = [(f"Flat {DEFAULT_RATE} km/day (today)", pd.Series(DEFAULT_RATE, index=test.index)),
             (f"Flat {g_med:.0f} km/day (median of first intervals, refreshed each run)", pd.Series(g_med, index=test.index)),
             ("Median by car age group", test["p_age"]), ("Median by make", test["p_make"]),
             ("Median by make + model", test["p_mm"]), ("Median by make + model + car age", test["p_mm_age"]),
             ("Car's own lifetime average (odometer / car age)", test["p_life"]),
             ("Lifetime average, adjusted by car age", test["p_life_adj"]),
             ("Average of adjusted lifetime and make + model + age", test["p_blend"])]
    default_test = pd.DataFrame([{"Default used for a car with no history": n, "Cars tested": len(test),
                                  **score(p, test["rate"], test["days"])} for n, p in cands])
    return {"explained": explained, "overall": overall, "by_make": by_make, "by_model": by_model, "extremes": extremes,
            "by_age": by_age, "default_test": default_test, "n_train": len(train), "priors_source": fi}


# ---------------------------------------------------------------- 2. long-absent cars
GAP_BINS = [0, 91, 182, 273, 365, 10000]
GAP_LABELS = ["under 3 months", "3-6 months", "6-9 months", "9-12 months", "over 12 months"]


def november_visitors(model: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """First visit per plate after the model's data ended, with the model's own prediction for that day."""
    first = (inv.dropna(subset=["plate_key"]).sort_values(["plate_key", "InvoiceDate", "InvoiceNumber"])
             .drop_duplicates("plate_key"))
    cand = first[["plate_key", "mobile_key"]].merge(model, on="plate_key", how="inner")
    cand["exact_id"] = cand["mobile_key"] == cand["model_mobile"]
    cand = (cand.sort_values(["plate_key", "exact_id", "customer_last_transaction_dt"], ascending=[True, False, False])
            .drop_duplicates("plate_key").drop(columns="mobile_key"))
    v = first.merge(cand, on="plate_key", how="inner")
    v = v[v["plate_mobiles"] <= JUNK_PLATE_MOBILES]
    v["days"] = (v["InvoiceDate"].dt.normalize() - v["customer_last_transaction_dt"].dt.normalize()).dt.days
    v = v[(v["days"] >= 1) & v["customer_last_current_mileage"].notna() & (v["actual_mileage"] > 0)].copy()
    v["actual_rate"] = (v["actual_mileage"] - v["customer_last_current_mileage"]) / v["days"]
    v = v[(v["actual_rate"] >= 0) & (v["actual_rate"] <= MAX_RATE)].copy()
    v["pred_rate"] = v["customer_avg_mileage_per_day"]
    v["err_km"] = (v["pred_rate"] - v["actual_rate"]) * v["days"]          # + = model predicted too many km
    v["gap"] = pd.cut(v["days"], GAP_BINS, labels=GAP_LABELS, right=False)
    v["source"] = np.where(v["pred_rate"] == DEFAULT_RATE, "default 55", "own history")
    return v


def section_long_absent(nv: pd.DataFrame) -> dict:
    own = nv[nv["source"] == "own history"]

    def dist(df):
        rows = []
        for gap, g in df.groupby("gap", observed=True):
            e = g["err_km"]
            worst = e.abs().nlargest(max(len(e) // 10, 1)).sum() / e.abs().sum()
            lo, hi = e.quantile(.05), e.quantile(.95)
            rows.append({"Time since last visit": gap, "Cars": len(g), "Mean error km": e.mean(),
                         "Mean without the 5% extremes each side": e[(e >= lo) & (e <= hi)].mean(), **pct_table(e),
                         "% model too high": (e > 0).mean(), "% too high by 2,500+ km": (e > 2500).mean(),
                         "% too low by 2,500+ km": (e < -2500).mean(), "% within 2,500 km": (e.abs() <= 2500).mean(),
                         "% of total error from the worst 10% of cars": worst,
                         "Median predicted km/day": g["pred_rate"].median(), "Median actual km/day": g["actual_rate"].median(),
                         "Median actual / predicted km/day": (g["actual_rate"] / g["pred_rate"]).median()})
        return pd.DataFrame(rows)

    edges = np.arange(-15000, 15001, 1000)
    mids = [f"{int(a / 1000):+d}k to {int(b / 1000):+d}k" for a, b in zip(edges[:-1], edges[1:])]
    curve = pd.DataFrame({"Error (predicted - actual km)": ["below -15k", *mids, "above +15k"]})
    for gap, g in own.groupby("gap", observed=True):
        h = np.histogram(g["err_km"], bins=[-np.inf, *edges, np.inf])[0]
        curve[str(gap)] = h / h.sum()

    # what a correction by time away would do: scale km/day by the median actual/predicted of each gap band
    factor = (own["actual_rate"] / own["pred_rate"]).groupby(own["gap"], observed=True).median()
    adj = own.assign(pred2=own["pred_rate"] * own["gap"].map(factor).astype(float))
    adj["err2"] = (adj["pred2"] - adj["actual_rate"]) * adj["days"]
    fix = adj.groupby("gap", observed=True).apply(lambda g: pd.Series({
        "Cars": len(g), "km/day multiplier": factor[g.name],
        "Median abs error km, today": g["err_km"].abs().median(), "Median abs error km, corrected": g["err2"].abs().median(),
        "% within 2,500 km, today": (g["err_km"].abs() <= 2500).mean(), "% within 2,500 km, corrected": (g["err2"].abs() <= 2500).mean(),
        "Median error km, today": g["err_km"].median(), "Median error km, corrected": g["err2"].median()}),
        include_groups=False).reset_index().rename(columns={"gap": "Time since last visit"})
    return {"own": dist(own), "default": dist(nv[nv["source"] == "default 55"]), "curve": curve, "fix": fix}


# ---------------------------------------------------------------- 4. plate vs plate + mobile
SWITCH_BINS = [0, 91, 182, 365, 100000]
SWITCH_LABELS = ["under 3 months", "3-6 months", "6-12 months", "over 12 months"]


def section_plate(v: pd.DataFrame, o: pd.DataFrame, cars: pd.DataFrame, model: pd.DataFrame) -> dict:
    multi = v[v["plate_mobiles"] >= 2].copy()
    multi["switch"] = (multi["mobile"] != multi.groupby("plate")["mobile"].shift()) & multi.groupby("plate").cumcount().gt(0)
    per = multi.groupby("plate").agg(mobiles=("mobile", "nunique"), switches=("switch", "sum"))
    plates = v.groupby("plate")["plate_mobiles"].first()
    clean = per["switches"] == per["mobiles"] - 1
    overview = pd.DataFrame([
        {"Plates": "All plates in the history", "Count": len(plates)},
        {"Plates": "Always the same mobile", "Count": int((plates == 1).sum())},
        {"Plates": "Two or more mobiles", "Count": len(per)},
        {"Plates": "  - clean handover (old mobile never comes back)", "Count": int(clean.sum())},
        {"Plates": "  - mobiles alternate (family, drivers, shared car)", "Count": int((~clean).sum())}])
    overview["% of all plates"] = overview["Count"] / len(plates)

    # clean handovers between exactly two mobiles: A then B
    ab = multi[multi["plate"].isin(per.index[clean & (per["mobiles"] == 2)])]
    ids = ab.groupby(["plate", "_id"], sort=False).agg(first=("dt", "first"), last=("dt", "last"), visits=("dt", "size")).reset_index()
    ids = ids.sort_values(["plate", "first"])
    ids["pos"] = ids.groupby("plate").cumcount()
    a, b = ids[ids["pos"] == 0].set_index("plate"), ids[ids["pos"] == 1].set_index("plate")
    h = a.join(b, lsuffix="_a", rsuffix="_b")
    h["gap_days"] = (h["first_b"] - h["last_a"]).dt.days
    h["gap"] = pd.cut(h["gap_days"], SWITCH_BINS, labels=SWITCH_LABELS, right=False)
    cr = cars[["rate", "first_km", "last_km", "first_dt", "last_dt"]]
    h = h.join(cr.add_suffix("_ca"), on="_id_a").join(cr.add_suffix("_cb"), on="_id_b")
    h["ratio"] = h["rate_cb"] / h["rate_ca"]
    usual_gap = v.groupby("_id")["dt"].diff().dt.days.median()

    def ratio_stats(r, label, n_all):
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        return {"Cars": label, "Cars with km/day on both sides": len(r), "Median new / old km/day": r.median(),
                "P25": r.quantile(.25), "P75": r.quantile(.75), "% within 25% of the old km/day": r.between(0.75, 1.25).mean(),
                "% changed by more than 50%": (~r.between(0.5, 1.5)).mean()}

    # baseline: cars that never changed mobile, first half of their history vs second half
    s = o[o["_id"].isin(v.loc[v["plate_mobiles"] == 1, "_id"])].copy()
    s["n"] = s.groupby("_id")["km"].transform("size")
    s = s[s["n"] >= 4]
    s["half"] = (s["seq"] >= s["n"] / 2).astype(int)
    hh = s.groupby(["_id", "half"]).agg(k0=("km", "first"), k1=("km", "last"), d0=("dt", "first"), d1=("dt", "last"))
    hh["span"] = (hh["d1"] - hh["d0"]).dt.days
    hh["rate"] = np.where(hh["span"] >= 60, (hh["k1"] - hh["k0"]) / hh["span"], np.nan)
    hh.loc[~hh["rate"].between(0.01, MAX_RATE), "rate"] = np.nan
    w = hh["rate"].unstack("half").dropna()
    change = [ratio_stats(w[1] / w[0], "Baseline: same mobile throughout (2nd half vs 1st half of history)", len(w))]
    for gap, g in h.groupby("gap", observed=True):
        change.append(ratio_stats(g["ratio"], f"Mobile changed, gap at the change {gap}", len(g)))
    change.append(ratio_stats(h["ratio"], "Mobile changed, all gaps", len(h)))
    change = pd.DataFrame(change)

    handover = h.groupby("gap", observed=True).agg(
        Plates=("gap_days", "size"), median_gap_days=("gap_days", "median"),
        median_visits_before=("visits_a", "median"), median_visits_after=("visits_b", "median"),
        only_one_visit_after=("visits_b", lambda s: (s == 1).mean()),
        odometer_went_backwards=("first_km_cb", lambda s: np.nan)).reset_index()
    back = h.dropna(subset=["first_km_cb", "last_km_ca"])
    handover["odometer_went_backwards"] = handover["gap"].map(
        (back["first_km_cb"] < back["last_km_ca"]).groupby(back["gap"], observed=True).mean()).astype(float)
    handover = handover.rename(columns={
        "gap": "Gap between last visit on old mobile and first on new", "median_gap_days": "Median gap days",
        "median_visits_before": "Median visits before", "median_visits_after": "Median visits after",
        "only_one_visit_after": "% with only one visit on the new mobile",
        "odometer_went_backwards": "% odometer lower than before (not the same car, or typo)"})
    handover["% of handovers"] = handover["Plates"] / handover["Plates"].sum()

    # prediction test: the first interval on the new mobile (the model uses 55 there today)
    fb = o[(o["seq"] == 1) & o["ok"]].set_index("_id")[["rate", "days", "prev_km", "prev_dt"]].add_prefix("t_")
    t = h.join(fb, on="_id_b").dropna(subset=["t_rate"])
    bd = (t["t_prev_dt"] - t["last_dt_ca"]).dt.days
    t["bridge"] = np.where(bd >= 30, (t["t_prev_km"] - t["last_km_ca"]) / bd, np.nan)
    ad = (t["t_prev_dt"] - t["first_dt_ca"]).dt.days
    t["plate_all"] = np.where(ad >= 60, (t["t_prev_km"] - t["first_km_ca"]) / ad, np.nan)
    for c in ("bridge", "plate_all"):
        t.loc[~t[c].between(0.01, MAX_RATE), c] = np.nan
    rows = []
    for gap, g in [*t.groupby("gap", observed=True), ("all gaps", t)]:
        g = g.dropna(subset=["rate_ca", "bridge", "plate_all"])
        for name, p in [(f"Flat {DEFAULT_RATE} km/day (today: history starts again)", pd.Series(DEFAULT_RATE, index=g.index)),
                        ("km/day under the old mobile", g["rate_ca"]),
                        ("km/day across the change (old last visit -> new first visit)", g["bridge"]),
                        ("km/day over the whole plate history", g["plate_all"])]:
            rows.append({"Gap at the mobile change": gap, "Prediction for the first interval on the new mobile": name,
                         "Cars tested": len(g), **score(p, g["t_rate"], g["t_days"])})
    pred = pd.DataFrame(rows)

    # how many of today's cars this touches
    act = model[model["is_active"]].copy()
    plate_has_rate = cars["rate"].notna().groupby(cars.index.str.split("__").str[0]).sum()
    act["other_ids_with_rate"] = act["_id"].str.split("__").str[0].map(plate_has_rate).fillna(0) - act["_id"].map(cars["rate"].notna()).fillna(False).astype(int)
    is_def = act["customer_avg_mileage_per_day"] == DEFAULT_RATE
    scope = pd.DataFrame([
        {"Active cars in the model": "All", "Cars": len(act)},
        {"Active cars in the model": f"On the flat {DEFAULT_RATE} km/day", "Cars": int(is_def.sum())},
        {"Active cars in the model": f"On the flat {DEFAULT_RATE} km/day, and the plate has km/day history under another mobile",
         "Cars": int((is_def & (act["other_ids_with_rate"] > 0) & (act["plate_mobiles"] <= JUNK_PLATE_MOBILES)).sum())}])
    scope["% of active cars"] = scope["Cars"] / len(act)
    return {"overview": overview, "handover": handover, "change": change, "pred": pred, "scope": scope, "usual_gap": usual_gap}


# ---------------------------------------------------------------- 5. time-based rule
RATIO_BINS = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, np.inf]
RATIO_LABELS = ["under 50%", "50-80%", "80-100%", "100-120%", "120-150%", "150-200%", "200-300%", "over 300%"]


def oil_history(v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oil = v[v["oil"] == True].sort_values(["_id", "dt"]).copy()  # noqa: E712
    g = oil.groupby("_id", sort=False)
    oil["prev_dt"], oil["prev_km"], oil["prev_syn"] = g["dt"].shift(), g["km"].shift(), g["syn"].shift()
    oil["days"] = (oil["dt"] - oil["prev_dt"]).dt.days
    oil["dkm"] = oil["km"] - oil["prev_km"]
    it = oil[(oil["days"] >= 14)].copy()
    it["oil_type"] = np.where(it["prev_syn"] == True, "synthetic (10,000 km rule)", "mineral (5,000 km rule)")  # noqa: E712
    it["rule_km"] = np.where(it["prev_syn"] == True, 10000, 5000)  # noqa: E712
    it.loc[~((it["dkm"] > 0) & (it["dkm"] / it["days"] <= MAX_RATE)), "dkm"] = np.nan
    per = oil.groupby("_id", sort=False).agg(last_oil_dt=("dt", "last"), last_syn=("syn", "last"))
    per = per.join(it.groupby("_id").agg(own_days=("days", "median"), own_km=("dkm", "median"), oil_intervals=("days", "size")))
    return it, per


def section_time_rule(it: pd.DataFrame) -> dict:
    rows = []
    for name, g in it.groupby("oil_type"):
        k = g["dkm"].dropna()
        rule = g["rule_km"].iloc[0]
        rows.append({"Oil at the previous change": name, "Oil changes": len(g), "Measure": "km driven between oil changes", **pct_table(k),
                     "% came before the rule km": (k < rule).mean(), "% came before 80% of the rule km": (k < 0.8 * rule).mean(),
                     "% came after 150% of the rule km": (k > 1.5 * rule).mean()})
        rows.append({"Oil at the previous change": name, "Oil changes": len(g), "Measure": "days between oil changes", **pct_table(g["days"])})
    intervals = pd.DataFrame(rows)

    # is a customer more regular in days or in km?
    full = it.dropna(subset=["dkm"]).copy()
    full["n"] = full.groupby("_id")["days"].transform("size")
    full = full[full["n"] >= 3]
    cv = full.groupby("_id").agg(cv_days=("days", lambda s: s.std() / s.mean()), cv_km=("dkm", lambda s: s.std() / s.mean()))
    regular = pd.DataFrame([{"Cars with 3+ oil intervals": len(cv), "Typical variation in days between changes": cv["cv_days"].median(),
                             "Typical variation in km between changes": cv["cv_km"].median(),
                             "% of cars more regular in days than in km": (cv["cv_days"] < cv["cv_km"]).mean()}])

    # predict the date of the car's latest oil change from what was known before it
    full = full.sort_values(["_id", "dt"])
    full["rank_rev"] = full.groupby("_id").cumcount(ascending=False)
    tgt, hist = full[full["rank_rev"] == 0].set_index("_id"), full[full["rank_rev"] > 0]
    hs = hist.groupby("_id").agg(med_days=("days", "median"), med_km=("dkm", "median"), h_km=("dkm", "sum"), h_d=("days", "sum"))
    hs["rate"] = (hs["h_km"] / hs["h_d"]).clip(lower=5)
    t = tgt.join(hs, how="inner")
    preds = {"km rule as in the model (5,000 / 10,000 km at the car's km/day)": t["rule_km"] / t["rate"],
             "km rule with the car's own usual km between changes": t["med_km"] / t["rate"],
             "time rule: the car's own usual days between changes": t["med_days"],
             "whichever comes first: model km rule or own usual days": np.minimum(t["rule_km"] / t["rate"], t["med_days"])}
    timing = pd.DataFrame([{"Rule for 'when is the next oil change'": n, "Cars tested": len(t),
                            "Median miss in days": (p - t["days"]).abs().median(),
                            "Median error days (+ = rule says later than they came)": (p - t["days"]).median(),
                            "% within 30 days": ((p - t["days"]).abs() <= 30).mean(),
                            "% rule fires after the customer already came (30+ days late)": ((p - t["days"]) > 30).mean(),
                            "% rule fires 30+ days too early": ((p - t["days"]) < -30).mean()} for n, p in preds.items()])
    return {"intervals": intervals, "regular": regular, "timing": timing}


def section_replay(model: pd.DataFrame, per: pd.DataFrame, inv: pd.DataFrame, month: pd.DataFrame, target_month: str,
                   list_date: pd.Timestamp, recent_since: pd.Timestamp) -> dict:
    """What a time rule would have added to the month's list, and how many of those customers came anyway."""
    p = model[model["is_active"] & (model["plate_mobiles"] <= JUNK_PLATE_MOBILES)].merge(per, left_on="_id", right_index=True, how="left")
    on = lambda mobs, pls: p["model_mobile"].isin(mobs) | p["plate_key"].isin(pls)  # noqa: E731
    keys = lambda d: (set(d["mobile_key"].dropna()), set(d["plate_key"].dropna()))  # noqa: E731
    p["listed"] = on(*keys(month))
    removed = on(*keys(inv[(inv["InvoiceDate"] >= recent_since) & (inv["InvoiceDate"] < list_date)]))
    for mobs, pls in vf.cleaning_sets(target_month).values():
        removed |= on(mobs, pls)
    p = p[~removed].copy()
    p["came"] = on(*keys(inv[inv["InvoiceDate"] >= list_date]))

    pop_days = p.loc[p["oil_intervals"] >= 2].groupby("last_syn")["own_days"].median()
    p["usual_days"] = np.where(p["oil_intervals"] >= 2, p["own_days"], p["last_syn"].map(pop_days))
    p["days_since_oil"] = (list_date - p["last_oil_dt"]).dt.days
    p["time_ratio"] = p["days_since_oil"] / p["usual_days"]
    p["km_ratio"] = p["pred_km_since_oil"] / p["oil_interval"]
    p["time_band"] = pd.cut(p["time_ratio"], RATIO_BINS, labels=RATIO_LABELS, right=False)
    p["km_band"] = pd.cut(p["km_ratio"], RATIO_BINS, labels=RATIO_LABELS, right=False)
    has = p.dropna(subset=["time_band", "km_band"])

    def curve(col, label):
        t = has.groupby(col, observed=True).agg(Cars=("came", "size"), Came=("came", "sum")).reset_index().rename(columns={col: label})
        t["% came"] = t["Came"] / t["Cars"]
        return t

    time_curve = curve("time_band", "Days since last oil change, as % of the car's usual days between changes")
    km_curve = curve("km_band", "Predicted km since last oil change, as % of the 5,000 / 10,000 km rule")
    grid = has.pivot_table(index="time_band", columns="km_band", values="came", aggfunc="mean", observed=True)
    grid_n = has.pivot_table(index="time_band", columns="km_band", values="came", aggfunc="size", observed=True)
    for gdf in (grid, grid_n):
        gdf.index.name = "Time (rows) x km (columns)"
        gdf.columns = [str(c) for c in gdf.columns]

    # candidate rules on customers NOT on the month's list, one message per mobile
    u = has[~has["listed"]].copy()
    own = u["oil_intervals"] >= 2
    months = (list_date - u["customer_last_transaction_dt"]).dt.days / 30.44
    km_overdue = (u["floor_now"] >= 1) & (u["floor_now"] == u["floor_prev"])
    rules = [("km overdue, any age (the rule from before)", km_overdue),
             ("km overdue, last visit under 6 months ago", km_overdue & (months < 6)),
             ("Time: 80-150% of own usual days (cars with 2+ oil intervals)", own & u["time_ratio"].between(0.8, 1.5)),
             ("Time: 90-130% of own usual days (cars with 2+ oil intervals)", own & u["time_ratio"].between(0.9, 1.3)),
             ("Time: 80-150% of usual days (own, or the average where no history)", u["time_ratio"].between(0.8, 1.5)),
             ("Time 80-150% of own usual days AND km overdue", own & u["time_ratio"].between(0.8, 1.5) & km_overdue),
             ("Time 80-150% of own usual days, NOT km overdue (only the time rule finds them)", own & u["time_ratio"].between(0.8, 1.5) & ~km_overdue),
             ("km overdue under 6 months OR time 80-150% of own usual days", (km_overdue & (months < 6)) | (own & u["time_ratio"].between(0.8, 1.5)))]
    rows = []
    for name, mask in rules:
        s = u[mask].sort_values("came", ascending=False).drop_duplicates("model_mobile")
        rows.append({"Rule for adding customers who were not on the list": name, "Extra messages": len(s),
                     "Came in the window": int(s["came"].sum()), "% came": s["came"].mean() if len(s) else np.nan})
    listed = has[has["listed"]].sort_values("came", ascending=False).drop_duplicates("model_mobile")
    rows.append({"Rule for adding customers who were not on the list": "For comparison: customers on the month's list (after cleaning)",
                 "Extra messages": len(listed), "Came in the window": int(listed["came"].sum()), "% came": listed["came"].mean()})
    # same list size, better picked: score cells on one random half, pick the top customers in the other half
    rng = np.random.default_rng(11)
    one = has.sort_values("came", ascending=False).drop_duplicates("model_mobile")
    a_half = rng.random(len(one)) < 0.5
    cell = one[a_half].groupby(["time_band", "km_band"], observed=True)["came"].mean().rename("cell_rate").reset_index()
    bh = one[~a_half].merge(cell, on=["time_band", "km_band"], how="left").sort_values("cell_rate", ascending=False)
    k = int(bh["listed"].sum())
    top = bh.head(k)
    rows.append({"Rule for adding customers who were not on the list": "Same size as the list, picked by the time x km grid instead (tested on a held-out half, scaled x2)",
                 "Extra messages": 2 * k, "Came in the window": 2 * int(top["came"].sum()), "% came": top["came"].mean()})
    rows.append({"Rule for adding customers who were not on the list": "  ... of which already on the month's list",
                 "Extra messages": 2 * int(top["listed"].sum()), "Came in the window": 2 * int(top.loc[top["listed"], "came"].sum()),
                 "% came": top.loc[top["listed"], "came"].mean()})
    return {"time_curve": time_curve, "km_curve": km_curve, "grid": grid.reset_index(), "grid_n": grid_n.reset_index(),
            "rules": pd.DataFrame(rows), "n": len(has)}


# ---------------------------------------------------------------- excel
def write_excel(path: Path, sheets: dict[str, list]) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        wb = xw.book
        f_title = wb.add_format({"bold": True, "font_size": 13})
        f_note = wb.add_format({"italic": True, "font_color": "#555555", "text_wrap": True, "valign": "top"})
        f_head = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "text_wrap": True, "valign": "top", "border": 1})
        f_int, f_dec = wb.add_format({"num_format": "#,##0"}), wb.add_format({"num_format": "#,##0.00"})
        f_pct, f_txt = wb.add_format({"num_format": "0.0%"}), wb.add_format({"text_wrap": True})
        for name, blocks in sheets.items():
            ws = wb.add_worksheet(name)
            xw.sheets[name] = ws
            ws.set_column(0, 0, 58)
            ws.set_column(1, 30, 15)
            r = 0
            for block in blocks:
                title, note, df = block[0], block[1], block[2]
                ws.write(r, 0, title, f_title)
                r += 1
                if note:
                    ws.merge_range(r, 0, r, 9, note, f_note)
                    ws.set_row(r, 15 * (1 + len(note) // 150))
                    r += 1
                if df is None:
                    r += 1
                    continue
                for j, c in enumerate(df.columns):
                    ws.write(r, j, str(c), f_head)
                ws.set_row(r, 48)
                top = r
                for i, row in enumerate(df.itertuples(index=False), start=r + 1):
                    for j, val in enumerate(row):
                        col = str(df.columns[j])
                        if isinstance(val, str) or val is None:
                            ws.write(i, j, val or "", f_txt)
                        elif pd.isna(val):
                            ws.write_blank(i, j, None)
                        elif col.startswith("%") or block[3:] == ("pct",) and j > 0:
                            ws.write_number(i, j, float(val), f_pct)
                        elif float(val) == int(val) and abs(val) >= 100 or col in ("Cars", "Count", "Plates", "Groups"):
                            ws.write_number(i, j, float(val), f_int)
                        else:
                            ws.write_number(i, j, float(val), f_dec if abs(val) < 100 else f_int)
                r += len(df) + 1
                if block[3:] == ("pct", "chart") or block[3:] == ("chart",):
                    ch = wb.add_chart({"type": "line"})
                    for j in range(1, len(df.columns)):
                        ch.add_series({"name": [name, top, j], "categories": [name, top + 1, 0, top + len(df), 0],
                                       "values": [name, top + 1, j, top + len(df), j], "smooth": True})
                    ch.set_title({"name": title})
                    ch.set_y_axis({"num_format": "0%", "name": "share of cars"})
                    ch.set_x_axis({"name": df.columns[0], "num_font": {"rotation": -60, "size": 8}})
                    ch.set_size({"width": 1050, "height": 420})
                    ws.insert_chart(r + 1, 0, ch)
                    r += 23
                r += 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month-list", required=True, type=Path)
    ap.add_argument("--target-month", required=True)
    ap.add_argument("--list-date", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--recent-since", default=None, help="start of the cleaning's recent-sales window (default: model cutoff + 7 days)")
    ap.add_argument("--rebuild-cache", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if a.rebuild_cache or not CACHE.exists():
        build_cache()

    snapshot = pd.Timestamp(f"1 {a.target_month}") + pd.offsets.MonthEnd(0)
    list_date = pd.Timestamp(a.list_date)
    v = load_visits()
    o = odo_intervals(v)
    cars = car_rates(o)
    veh = vehicles(snapshot)
    print(f"history: {len(v):,} visits, {v['_id'].nunique():,} cars, {v['dt'].min().date()} .. {v['dt'].max().date()}")

    model = vf.load_model(snapshot, list_date.date())
    cutoff = model["customer_last_transaction_dt"].max().normalize()
    inv = vf.fetch_invoices((cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), a.end)
    month = vf.load_list(a.month_list)
    recent_since = pd.Timestamp(a.recent_since) if a.recent_since else cutoff + pd.Timedelta(days=7)

    s1 = section_make_model(o, cars, veh)
    nv = november_visitors(model, inv)
    s2 = section_long_absent(nv)
    s4 = section_plate(v, o, cars, model)
    it, per = oil_history(v)
    s5 = section_time_rule(it)
    s6 = section_replay(model, per, inv, month, a.target_month, list_date, recent_since)

    win = f"{list_date:%d %b} to {pd.Timestamp(a.end):%d %b %Y}"
    readme = pd.DataFrame({"Sheet": ["1 Make and model", "1 Default km per day", "2 Long-absent cars", "4 Plate history", "5 Oil timing", "5 Time rule on the list"],
                           "Question": ["Is km/day tied to the make and model of the car?",
                                        f"For a car with no history, what works better than the flat {DEFAULT_RATE} km/day?",
                                        "Is the over-prediction for long-absent cars typical, or caused by a few extreme cars?",
                                        "When a plate comes back under a new mobile, does its old history still describe it?",
                                        "Do customers change oil on km or on time?",
                                        "What would a time rule have added to the month's list, and at what hit rate?"]})
    sheets = {
        "README": [("Model improvement evidence", f"History: {len(v):,} visits of {v['_id'].nunique():,} cars, {v['dt'].min():%d %b %Y} to {v['dt'].max():%d %b %Y} "
                    f"(the data the {a.target_month} model was built on). Visits after it: MACDB M-invoices to {pd.Timestamp(a.end):%d %b %Y}. "
                    f"km/day above {MAX_RATE} or below 0 is treated as an odometer typo and left out, as the model does.", readme)],
        "1 Make and model": [
            ("How much of the difference between cars each attribute explains", "km/day over each car's whole history. Scored on cars not used to fit the group averages. "
             "0% = knowing it tells nothing, 100% = it fixes the car's km/day exactly.", s1["explained"]),
            ("All cars", None, s1["overall"]), ("By car age", "km/day: 25% of cars are below P25, half below the median, 75% below P75.", s1["by_age"]),
            ("25 most common makes", None, s1["by_make"]), ("30 most common models", None, s1["by_model"]),
            ("Models with the highest and lowest median (2,000+ cars)", None, s1["extremes"])],
        "1 Default km per day": [
            ("Which default predicts a new car's first interval best", f"Test: the km/day between a car's 1st and 2nd visit, which is where the model has no history and uses {DEFAULT_RATE}. "
             f"Defaults were learned from {s1['n_train']:,} first intervals ending before {SPLIT:%d %b %Y} and tested on later ones. "
             "Lifetime average = odometer at the first visit / age of the car.", s1["default_test"])],
        "2 Long-absent cars": [
            ("Error by time since last visit, cars with their own km/day history", f"First visit per car after the model's data ended. Error = predicted km - actual km at the visit; "
             "+ means the model predicted more driving than happened. P50 is the median car.", s2["own"]),
            ("Shape of the error", "Share of cars in each 1,000 km error band. A curve shifted right = most cars over-predicted; a centred curve with a long tail = a few extreme cars.",
             s2["curve"], "pct", "chart"),
            ("Effect of scaling km/day by time away", "km/day multiplied by the median actual / predicted ratio of the band (fitted and tested on the same cars, so slightly flattering).", s2["fix"]),
            (f"Same error table, cars on the flat {DEFAULT_RATE} km/day", None, s2["default"])],
        "4 Plate history": [
            ("How many plates change mobile", "Plates used by more than 10 mobiles are placeholders and are left out.", s4["overview"]),
            ("Clean handovers between two mobiles", f"For comparison, the usual gap between two visits of the same car-customer is {s4['usual_gap']:.0f} days.", s4["handover"]),
            ("Did the driving change with the mobile?", "km/day on the new mobile / km/day on the old one. The baseline shows how much km/day drifts over time without any change of mobile.", s4["change"]),
            ("Which history predicts the first interval on the new mobile", "Test: km/day between the 1st and 2nd visit on the new mobile. Same cars for all four options.", s4["pred"]),
            ("How many of today's cars this affects", None, s4["scope"])],
        "5 Oil timing": [
            ("Between two oil changes: km driven and days passed", "From the history; only cars that came back for oil.", s5["intervals"]),
            ("Is a customer more regular in days or in km?", "Variation = standard deviation / mean of the car's own intervals; lower = more regular.", s5["regular"]),
            ("Which rule predicts the date of the next oil change", "Test: the car's latest oil change, predicted from its earlier ones (3+ oil intervals).", s5["timing"])],
        "5 Time rule on the list": [
            ("Who came, by time since the last oil change", f"Active cars in the model after the wave cleaning ({s6['n']:,} with oil history). Came = an M-invoice {win} on the mobile or plate. "
             "Nobody had been messaged in most of this window, so this is who comes anyway.", s6["time_curve"]),
            ("Who came, by predicted km since the last oil change", None, s6["km_curve"]),
            ("% came: time (rows) x km (columns)", None, s6["grid"], "pct"), ("Cars in each cell", None, s6["grid_n"]),
            ("Cost of each rule", "Customers not on the month's list that the rule would add, one message per mobile.", s6["rules"])],
    }
    out_dir = vf.OUTPUTS / pd.Timestamp(f"1 {a.target_month}").strftime("%b-%Y").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"improvement_analysis_{a.target_month.lower().replace(' ', '_')}.xlsx"
    write_excel(out, sheets)
    with pd.option_context("display.width", 260, "display.max_colwidth", 90, "display.max_rows", 200, "display.max_columns", 30):
        for name, blocks in sheets.items():
            for b in blocks:
                print(f"\n### {name} | {b[0]}")
                if b[2] is not None:
                    print(b[2].round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
