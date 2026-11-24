"""
Back-test: today's oil-due rule vs ranking customers by time x km since their last oil change.

Replays past months from the visit history the model is built on (post-processing/final/btl_ranking.py,
which also holds the feature code), no pipeline run needed. For each test month M:

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
sys.path.insert(0, str(HERE.parent / "final"))
from btl_ranking import features, load_visits, month_dates as dates  # noqa: E402

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
        train = train[~train["removed"]]
        cell = train.groupby(["time_band", "km_band"])["came"].mean().rename("cell")
        test = feats[m][~feats[m]["removed"]].join(cell, on=["time_band", "km_band"])
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
        tot = {c: r[c].sum() for c in r.columns if c.endswith(" came") or c == "Current list"}
        print(f"  current list {tot['Current list']:,} msgs -> {tot['Current came']:,} came ({tot['Current came'] / tot['Current list']:.2%})")
        print(f"  ranked same  {tot['Current list']:,} msgs -> {tot['Ranked, same size came']:,} came "
              f"({tot['Ranked, same size came'] / tot['Current list']:.2%})")
        print(f"  ranked 80%   {int(0.8 * tot['Current list']):,} msgs -> {tot['Ranked, 80% size came']:,} came")


if __name__ == "__main__":
    main()
