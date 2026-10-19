"""Turn the reviewed promos workbook back into the promo list the pipeline reads.

Output has the same format as the old `raw_promos.parquet` (one string column `Promo`): the old list's rows
first, in their original order, then every package marked `is_promotion == 1` in the review that isn't
already in it. Old promos are always kept, since some were renamed/deleted in MACDB but still match
historical sales.

Names are stripped of leading/trailing spaces: `columns_formatting` applies trim(upper()) to package_name
but `create_special_trx_features` only upper()s the promo list, so a promo with a trailing space never matched.

Usage (BTL env, from repo root):
    python scripts/build_promos_parquet.py data/01_raw/promos/promos_review_4.xlsx data/01_raw/raw_promos_2026_10.parquet
"""

import sys

import pandas as pd

OLD_PROMOS = "data/01_raw/raw_promos.parquet"


def main(review_path: str, out_path: str):
    old = pd.read_parquet(OLD_PROMOS)["Promo"].str.strip(" ")
    review = pd.read_excel(review_path, sheet_name="packages_review")

    unlabelled = review["is_promotion"].isna().sum()
    if unlabelled:
        raise ValueError(f"{unlabelled} packages in {review_path} have no is_promotion label")

    new = review.loc[review["is_promotion"].eq(1), "ServicePackageName"].astype(str).str.strip(" ")
    new = new[~new.str.upper().isin(old.str.upper())].drop_duplicates()

    out = pd.DataFrame({"Promo": pd.concat([old, new], ignore_index=True)})
    out.to_parquet(out_path, index=False)
    print(f"{out_path}: {len(old)} old + {len(new)} new = {len(out)} rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])
