"""Build an Excel workbook for manually reviewing which service packages are promotions.

`ingest_promos()` read MACDB.dbo.TMP_PROMOS, which we no longer have access to, so
`data/01_raw/raw_promos.parquet` is a frozen snapshot. The pipeline matches it against
`upper(package_name)` of transactions, i.e. the values are *service package names*.

This script pulls the package catalog (t_ServicePackage) and t_Promotion from MACDB, labels
every package against the old snapshot and writes `data/01_raw/promos/promos_review.xlsx`:

- previous - promotion      : package name is in raw_promos.parquet
- previous - non promotion  : created on/before the cutoff but not in raw_promos.parquet
- NEW - to review           : created after the cutoff (newest package that made it into the
                              old list), so nobody has classified it yet

PMS packages are never promotions (none of the old list is PMS) and go to their own sheet.

Usage (BTL env, from repo root):  python scripts/build_promos_review.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion_pipeline import _get_conn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PROMOS = "data/01_raw/raw_promos.parquet"
TRANSACTIONS = "data/01_raw/transactions_origin.parquet"
OUT = Path("data/01_raw/promos/promos_review.xlsx")

STATUS_NEW = "NEW - to review"
STATUS_PROMO = "previous - promotion"
STATUS_NON_PROMO = "previous - non promotion"


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def _package_usage() -> pd.DataFrame:
    """Invoice count and first/last invoice date per package name, from transactions_origin."""
    table = pq.read_table(TRANSACTIONS, columns=["InvoiceID", "InvoiceDate", "ServicePackageName"])
    table = table.set_column(1, "InvoiceDate", pc.cast(pc.utf8_slice_codeunits(table["InvoiceDate"], 0, 10), "date32"))
    usage = table.group_by("ServicePackageName").aggregate(
        [("InvoiceID", "count_distinct"), ("InvoiceDate", "min"), ("InvoiceDate", "max")]
    ).to_pandas().rename(columns={
        "InvoiceID_count_distinct": "n_invoices", "InvoiceDate_min": "first_invoice", "InvoiceDate_max": "last_invoice",
    })
    usage["k"] = _norm(usage["ServicePackageName"])
    return usage.groupby("k").agg(
        n_invoices=("n_invoices", "sum"), first_invoice=("first_invoice", "min"), last_invoice=("last_invoice", "max")
    )


def main():
    conn = _get_conn()
    logger.info("Pull t_ServicePackage")
    packages = pd.read_sql(
        "SELECT p.ServicePackageID, p.ServicePackageName, t.ServicePackageTypeName, p.IsActive, p.CreationOn, p.ModifiedOn "
        "FROM MACDB.dbo.t_ServicePackage p "
        "LEFT JOIN MACDB.dbo.t_ServicePackageType t ON p.ServicePackageTypeID = t.ServicePackageTypeID",
        conn,
    )
    logger.info("Pull t_Promotion")
    promotions = pd.read_sql("SELECT * FROM MACDB.dbo.t_Promotion", conn)
    old = pd.read_parquet(RAW_PROMOS)

    packages["k"] = _norm(packages["ServicePackageName"])
    promotions["k"] = _norm(promotions["PromotionName"])
    old["k"] = _norm(old["Promo"])

    logger.info("Package usage from transactions_origin")
    packages = packages.join(_package_usage(), on="k")
    packages["n_invoices"] = packages["n_invoices"].fillna(0).astype(int)

    # Packages get re-created under the same name (e.g. "99 Mineral Oil": 2023 and 2026 rows), and the
    # pipeline matches on name, so judge "new" per name: first creation of that name or first sale, whichever is earlier
    packages["first_seen"] = pd.concat(
        [packages.groupby("k")["CreationOn"].transform("min"), pd.to_datetime(packages["first_invoice"])], axis=1
    ).min(axis=1)

    in_old = packages["k"].isin(old["k"])
    cutoff = packages.loc[in_old, "first_seen"].max()
    logger.info(f"Cutoff (newest package in old promo list): {cutoff}")

    packages["status"] = STATUS_NON_PROMO
    packages.loc[packages["first_seen"] > cutoff, "status"] = STATUS_NEW
    packages.loc[in_old, "status"] = STATUS_PROMO
    packages["name_also_in_t_Promotion"] = packages["k"].isin(promotions["k"])
    # Pre-filled answer column: the user edits this, it's what the new promo file gets built from
    packages["is_promotion"] = packages["status"].map({STATUS_PROMO: 1, STATUS_NON_PROMO: 0})

    is_pms = packages["ServicePackageTypeName"].eq("PMS")
    cols = [
        "status", "is_promotion", "ServicePackageName", "ServicePackageTypeName", "IsActive", "first_seen", "CreationOn",
        "ModifiedOn", "n_invoices", "first_invoice", "last_invoice", "name_also_in_t_Promotion", "ServicePackageID",
    ]
    status_order = {STATUS_NEW: 0, STATUS_PROMO: 1, STATUS_NON_PROMO: 2}
    review = (
        packages.loc[~is_pms]
        .sort_values(["status", "first_seen"], key=lambda s: s.map(status_order) if s.name == "status" else s, ascending=[True, False])
        [cols]
    )
    pms = packages.loc[is_pms].sort_values("CreationOn", ascending=False)[cols]

    old_check = old.drop_duplicates("k").assign(found_in_t_ServicePackage=lambda d: d["k"].isin(packages["k"]))
    old_check = old_check.sort_values("found_in_t_ServicePackage")[["Promo", "found_in_t_ServicePackage"]]

    promotions = promotions.drop(columns="k").sort_values("CreationOn", ascending=False)
    promotions["PromotionID"] = promotions["PromotionID"].astype(str)

    readme = pd.DataFrame({"note": [
        "Source: MACDB t_ServicePackage (+ t_ServicePackageType), t_Promotion; old list = data/01_raw/raw_promos.parquet.",
        "The pipeline flags a transaction as promo when upper(ServicePackageName) is in the promo list -> review PACKAGES, not promotions.",
        f"Cutoff = {cutoff:%Y-%m-%d %H:%M} = first_seen of the newest package already in the old list.",
        "first_seen = earliest of (first creation of that NAME in t_ServicePackage, first sale). Names get re-created, so status is by name.",
        f"'{STATUS_NEW}': created after the cutoff, never classified. Fill is_promotion with 1/0.",
        f"'{STATUS_PROMO}' / '{STATUS_NON_PROMO}': pre-filled from the old list; change is_promotion if you disagree.",
        "n_invoices / first_invoice / last_invoice: usage in transactions_origin.parquet (0 = never sold).",
        "name_also_in_t_Promotion: package name also exists as a promotion name in t_Promotion (a hint only).",
        "Sheet pms_packages: PMS-type packages; none of the old promos are PMS, so assumed non promotion.",
        "Sheet old_list_check: old promo names; those not found in t_ServicePackage were renamed/deleted (kept matching only historical sales).",
    ]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        review.to_excel(writer, sheet_name="packages_review", index=False)
        pms.to_excel(writer, sheet_name="pms_packages", index=False)
        old_check.to_excel(writer, sheet_name="old_list_check", index=False)
        promotions.to_excel(writer, sheet_name="t_Promotion", index=False)
        for name, ws in writer.sheets.items():
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:500])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 70)

    logger.info(f"Wrote {OUT}")
    logger.info(f"packages_review status counts:\n{review['status'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
