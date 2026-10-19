"""Build the branch-coordinates override file from the ops "Search Engine" workbook.

MACDB v_Branch / v_PAC_Branch started returning NULL Latitude/Longitude (2026-07), which breaks
create_prm_geolocation. The ops team's Search Engine workbook has a Coordinates column per station (BE#,
which is BranchCode in MACDB). This script parses and validates those coordinates and writes
`data/01_raw/branch_coordinates.xlsx`, which create_prm_branches uses to override the DB values.

Sheets used: "PE Stations List" (current stations) and "Permanent Closed Stations" (closed ones still
needed for history). If a BE# is in both, the PE list wins; for closed BE#s listed more than once, the
latest Closing Date wins.

Validation per station (anything failing is left out of the override, so the DB value is used):
- parses as decimal "lat, lon" or DMS (24°36'58.1"N 46°38'59.0"E); lat/lon given in the wrong order is
  swapped back
- inside KSA: within KSA_MAX_KM of a cell of the WorldPop KSA 1km grid (land cells only, the same grid the
  model uses for population density)
Flags only (kept, but listed for review): far from the Aug-2025 DB coordinate, far from the station's
city median, coordinate shared with another BE#.

Output sheets: `coordinates` (what the pipeline reads) and `validation` (every row, every check).

Usage (BTL env, from repo root):  python scripts/build_branch_coordinates.py "search-engine/10 Search Engine-Oct '26.xlsx"
"""

import re
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

OUT = "data/01_raw/branch_coordinates.xlsx"
WORLD_POP = "data/01_raw/saudi_arabia/sau_pd_2020_1km_UNadj_ASCII_XYZ.csv"
OLD_DB_BRANCHES = "data/01_raw/branches.parquet"  # Aug-2025 snapshot, last time MACDB had coordinates

EARTH_KM = 6371.0
KSA_MAX_KM = 1.5
OLD_DB_FLAG_KM = 2.0
CITY_FLAG_KM = 50.0

# Checked by hand (Oct 2026): Search Engine coordinate is wrong, so it's left out of the override
REJECT = {
    "1289": "points at Hail city; Al Bukayriyah is ~26.14N 43.66E (Aug-2025 DB had 26.13, 43.61)",
    "1527": "copy of 2125 Umrah Land's coordinate (Aug-2025 DB had 21.453, 39.808)",
}
# Checked by hand (Oct 2026): plausible but couldn't confirm, kept
REVIEW = {
    "1879": "'Orange Tahlia Jubail' but the point is in Al Khobar (same in Aug-2025 DB)",
    "2318": "'Bani Sar Albaha' but the point is ~130 km south of Al Baha",
    "1804": "'AL SHIFA YANBU' but the point is in Madinah (same in Aug-2025 DB)",
    "1697": "differs from Aug-2025 DB by exactly 0.1 deg latitude (24.6175 vs 24.5175)",
}

DMS = re.compile(
    r"(\d+)\s*°\s*(\d+)\s*'\s*([\d.]+)\s*\"?\s*([NS])\s*,?\s*(\d+)\s*°\s*(\d+)\s*'\s*([\d.]+)\s*\"?\s*([EW])", re.I
)
DECIMAL = re.compile(r"(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)")


def parse_coordinates(text) -> tuple:
    """Return (lat, lon, note). lat/lon are NaN when the text can't be parsed."""
    if pd.isna(text):
        return np.nan, np.nan, "empty"
    text = str(text).strip()
    m = DMS.search(text)
    if m:
        d1, m1, s1, h1, d2, m2, s2, h2 = m.groups()
        a = (int(d1) + int(m1) / 60 + float(s1) / 3600) * (-1 if h1.upper() == "S" else 1)
        b = (int(d2) + int(m2) / 60 + float(s2) / 3600) * (-1 if h2.upper() == "W" else 1)
        return a, b, "DMS converted"
    m = DECIMAL.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        # KSA: lat 16-33, lon 34-56, so the two never overlap
        if a > 33.5 and b < 33.5:
            return b, a, "lat/lon swapped back"
        return a, b, ""
    return np.nan, np.nan, f"unparseable: {text}"


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def load_search_engine(path: str) -> pd.DataFrame:
    pe = pd.read_excel(path, sheet_name="PE Stations List", header=1)
    closed = pd.read_excel(path, sheet_name="Permanent Closed Stations", header=0)

    pe = pe.assign(sheet="PE Stations List", closing_date=pd.NaT)
    closed = closed.rename(columns={"Closing Date": "closing_date"}).assign(sheet="Permanent Closed Stations")
    closed = closed[closed["BE#"].notna()]
    closed = closed.sort_values("closing_date").drop_duplicates("BE#", keep="last")

    cols = ["BE#", "Station Name", "City", "Coordinates", "sheet", "closing_date"]
    df = pd.concat([pe[cols], closed[cols]], ignore_index=True)
    df["branch_code"] = pd.to_numeric(df["BE#"], errors="coerce").astype("Int64").astype(str)
    # PE list rows come first, so keep="first" lets the PE list win over the closed list
    return df.drop_duplicates("branch_code", keep="first").drop(columns="BE#")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["Coordinates"].apply(parse_coordinates)
    df["latitude"] = parsed.str[0]
    df["longitude"] = parsed.str[1]
    df["parse_note"] = parsed.str[2]
    has = df["latitude"].notna()

    world_pop = pd.read_csv(WORLD_POP, usecols=["X", "Y"])
    tree = BallTree(np.radians(world_pop[["Y", "X"]].to_numpy()), metric="haversine")
    dist, _ = tree.query(np.radians(df.loc[has, ["latitude", "longitude"]].to_numpy()), k=1)
    df.loc[has, "km_to_ksa_land"] = dist[:, 0] * EARTH_KM
    df["in_ksa"] = df["km_to_ksa_land"] <= KSA_MAX_KM

    old = pd.read_parquet(OLD_DB_BRANCHES, columns=["BranchCode", "Latitude", "Longitude"]).dropna()
    old = old[(old["Latitude"] != 0) | (old["Longitude"] != 0)]  # MACDB used (0, 0) as "unknown"
    old = old.drop_duplicates("BranchCode").set_index("BranchCode")
    df["old_db_lat"] = df["branch_code"].map(old["Latitude"])
    df["old_db_lon"] = df["branch_code"].map(old["Longitude"])
    df["km_from_old_db"] = _haversine_km(df["latitude"], df["longitude"], df["old_db_lat"], df["old_db_lon"])

    ok = df["in_ksa"]
    city = df["City"].astype(str).str.strip().str.upper()
    med = df[ok].groupby(city[ok])[["latitude", "longitude"]].median()
    df["km_from_city_median"] = _haversine_km(
        df["latitude"], df["longitude"], city.map(med["latitude"]), city.map(med["longitude"])
    )
    shared = df[has].groupby(["latitude", "longitude"])["branch_code"].transform(lambda s: ", ".join(s) if len(s) > 1 else "")
    df["shares_coordinate_with"] = shared.reindex(df.index).fillna("")

    df["status"] = np.select(
        [~has, ~df["in_ksa"], df["branch_code"].isin(REJECT.keys())],
        ["rejected: no coordinate", "rejected: outside KSA", "rejected: wrong (checked by hand)"],
        default="ok",
    )
    df["note"] = df["branch_code"].map({**REJECT, **REVIEW}).fillna("")
    flags = pd.DataFrame({
        f"> {OLD_DB_FLAG_KM:g} km from Aug-2025 DB": df["km_from_old_db"] > OLD_DB_FLAG_KM,
        f"> {CITY_FLAG_KM:g} km from city median": df["km_from_city_median"] > CITY_FLAG_KM,
        "shared coordinate": df["shares_coordinate_with"] != "",
        "format fixed": df["parse_note"].isin(["DMS converted", "lat/lon swapped back"]),
        "NEEDS REVIEW": df["branch_code"].isin(REVIEW.keys()),
    })
    df["flags"] = flags.apply(lambda r: "; ".join(r.index[r]), axis=1)
    return df


def main(search_engine_path: str):
    df = validate(load_search_engine(search_engine_path))

    coordinates = df.loc[df["status"] == "ok", ["branch_code", "latitude", "longitude", "Station Name", "sheet"]]
    coordinates = coordinates.rename(columns={"Station Name": "station_name", "sheet": "source"})
    validation = df[[
        "status", "flags", "note", "branch_code", "Station Name", "City", "sheet", "closing_date", "Coordinates", "parse_note",
        "latitude", "longitude", "km_to_ksa_land", "old_db_lat", "old_db_lon", "km_from_old_db",
        "km_from_city_median", "shares_coordinate_with",
    ]].sort_values(["status", "flags"], ascending=[False, False])

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        coordinates.to_excel(writer, sheet_name="coordinates", index=False)
        validation.to_excel(writer, sheet_name="validation", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

    print(f"{OUT}: {len(coordinates)} usable coordinates")
    print(validation["status"].value_counts().to_string())
    print(validation["flags"].str.split("; ").explode().replace("", np.nan).dropna().value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1])
