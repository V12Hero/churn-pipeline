"""Output layout shared by the BTL post-processing scripts.

    post-processing/final/outputs/
        exclusion_log.csv            every exclusion list ever applied (all months)
        nov-2026/                    everything produced for the Nov 2026 list
            BTL_Targetlist_Nov 2026.csv
            nov_2026_wave_1_2026-11-10.csv
            nov_2026_wave_1_MASTER_2026-11-10.csv
            ...

`outputs/` holds customer mobiles — it is git-ignored; never commit it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FINAL_DIR = Path(__file__).resolve().parent
REPO = FINAL_DIR.parents[1]
OUTPUTS = FINAL_DIR / "outputs"
EXCLUSION_LOG = OUTPUTS / "exclusion_log.csv"


def month_slug(target_month: str) -> str:
    """'Nov 2026' -> 'nov-2026'."""
    return pd.Timestamp(f"1 {target_month}").strftime("%b-%Y").lower()


def month_dir(target_month: str) -> Path:
    d = OUTPUTS / month_slug(target_month)
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_month_dir(target_month: str | None) -> Path:
    """Folder for --month if given, else the most recently modified month folder."""
    if target_month:
        return month_dir(target_month)
    dirs = [d for d in OUTPUTS.glob("*-[0-9][0-9][0-9][0-9]") if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no month folders in {OUTPUTS}; run btl_targetlist_create.py first")
    return max(dirs, key=lambda d: d.stat().st_mtime)
