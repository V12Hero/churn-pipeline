import datetime
import pandas as pd
import numpy as np
import pyodbc

import os
import glob
import tqdm

import logging
from rich.logging import RichHandler

from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

SERVER = os.getenv("SERVER")
DATABASE = os.getenv("DATABASE")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

logging.basicConfig(
    format= "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[RichHandler()],
)

logger = logging.getLogger(__name__)


@contextmanager
def _db_conn():
    """
    Context manager that opens a fresh pyodbc connection, yields it, and
    guarantees close() on exit (even on exception). Use:

        with _db_conn() as conn:
            ...

    This replaces the old `_get_conn()` pattern where connections were held
    open across the lifetime of a function and could leak on errors.
    """
    connectionString = (
        f'DRIVER={{ODBC Driver 18 for SQL Server}};'
        f'SERVER={SERVER};DATABASE={DATABASE};'
        f'UID={USERNAME};PWD={PASSWORD};'
        f'TrustServerCertificate=yes;'
    )
    conn = pyodbc.connect(connectionString)
    try:
        yield conn
    finally:
        conn.close()


def _fetch_df(cursor, query, params=None):
    """Run a query and return a DataFrame. Centralises the cursor->DataFrame plumbing."""
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    cols = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    # Avoid building DataFrame from empty rows with no schema
    return pd.DataFrame.from_records(rows, columns=cols)


def _has_column(cursor, tablename, column):
    """Check if a column exists without pulling a row of data."""
    cursor.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo'
          AND TABLE_NAME = ?
          AND COLUMN_NAME = ?
        """,
        (tablename, column),
    )
    return cursor.fetchone() is not None


def _month_range(start, end):
    """Yield (year, month, month_start_date, next_month_start_date) tuples.

    Using half-open date ranges (>= start AND < next_start) lets SQL Server
    use any index on ModifiedOn. The old code used YEAR()/MONTH() in the
    WHERE clause, which forces a full table scan because the predicate is
    not sargable.
    """
    dates_list = pd.date_range(start, end, freq='1ME')
    for date in dates_list:
        month_start = datetime.date(date.year, date.month, 1)
        if date.month == 12:
            next_month_start = datetime.date(date.year + 1, 1, 1)
        else:
            next_month_start = datetime.date(date.year, date.month + 1, 1)
        yield date, month_start, next_month_start


def _get_all_data_from_table(conn, tablename):

    cursor = conn.cursor()

    if _has_column(cursor, tablename, "ModifiedOn"):
        cursor.execute(f"SELECT MIN(ModifiedOn) FROM MACDB.dbo.{tablename}")
        min_date = cursor.fetchone()[0]
        max_date = datetime.date.today()

        tables_result_list = []
        for date, m_start, m_end in tqdm.tqdm(list(_month_range(min_date, max_date))):
            df = _fetch_df(
                cursor,
                f"""
                SELECT *
                FROM MACDB.dbo.{tablename}
                WHERE ModifiedOn >= ? AND ModifiedOn < ?
                """,
                (m_start, m_end),
            )
            if not df.empty:
                tables_result_list.append(df)

        if not tables_result_list:
            # Preserve an empty DataFrame with the right columns rather than crashing
            result_df = _fetch_df(cursor, f"SELECT TOP 0 * FROM MACDB.dbo.{tablename}")
        else:
            result_df = pd.concat(tables_result_list, ignore_index=True)

    else:
        result_df = _fetch_df(cursor, f"SELECT * FROM MACDB.dbo.{tablename}")

    if result_df is None:
         raise ValueError("result_df is None. There is a problem with cursor")

    return result_df


def get_all_data_from_invoice_table(conn, tablename):

    cursor = conn.cursor()

    cursor.execute(f"SELECT MIN(ModifiedOn) FROM MACDB.dbo.{tablename}")
    min_date = cursor.fetchone()[0]
    max_date = datetime.date.today() + datetime.timedelta(days=31)

    print(min_date, " - ", max_date)

    monthly_df_list = []

    for date, m_start, m_end in tqdm.tqdm(list(_month_range(min_date, max_date))):
        invoice_df = _fetch_df(
            cursor,
            f"""
            SELECT *
            FROM MACDB.dbo.{tablename}
            WHERE ModifiedOn >= ? AND ModifiedOn < ?
            """,
            (m_start, m_end),
        )
        if not invoice_df.empty:
            monthly_df_list.append(invoice_df)
            # invoice_df.to_parquet(f"{tablename}/{date.year}{date.month:02}.parquet")

    out = pd.concat(monthly_df_list, axis=0, ignore_index=True)

    return out


def get_all_data_from_invoice_items_table(conn, tablename, table_items_name, filepath, min_date):

    cursor = conn.cursor()

    cursor.execute(f"SELECT MIN(ModifiedOn) FROM MACDB.dbo.{tablename}")
    rows = cursor.fetchall()

    # min_date = rows[0][0]
    max_date = datetime.date.today() + datetime.timedelta(days=31)

    print(min_date, " - ", max_date)

    monthly_df_list = []

    try:
        os.mkdir(filepath)
        print(f"Directory '{filepath}' created successfully.")
    except FileExistsError:
        print(f"Directory '{filepath}' already exists.")

    for date, m_start, m_end in tqdm.tqdm(list(_month_range(min_date, max_date))):
        # Direct join on InvoiceID; range predicate on ModifiedOn is sargable
        # so SQL can use an index instead of scanning every row.
        items_df = _fetch_df(
            cursor,
            f"""
            SELECT a.*
            FROM MACDB.dbo.{table_items_name} a
            INNER JOIN MACDB.dbo.{tablename} b
                ON a.InvoiceID = b.InvoiceID
            WHERE b.ModifiedOn >= ? AND b.ModifiedOn < ?
            """,
            (m_start, m_end),
        )
        if not items_df.empty:
            # monthly_df_list.append(items_df)
            items_df.to_parquet(f"{filepath}/{date.year}{date.month:02}.parquet", index=False)

    # out = pd.concat(monthly_df_list, axis=0)
    out = pd.read_parquet(filepath)

    return out


def ingest_promos():
    with _db_conn() as conn:
        promos_df = _get_all_data_from_table(conn, "TMP_PROMOS")

    out = promos_df

    return out


def ingest_customers():
    with _db_conn() as conn:
        customer_PE_df = _get_all_data_from_table(conn, "v_Customer")
        customer_PAC_df = _get_all_data_from_table(conn, "v_PAC_Customer")

    cols = customer_PE_df.columns

    customer_PE_df["StationBrand"] = "PE"
    customer_PAC_df["StationBrand"] = "PAC"

    customer_df = pd.concat([customer_PE_df, customer_PAC_df], axis=0).drop_duplicates(subset=cols)

    customer_df["Mobile"] = customer_df["Mobile"].str.zfill(10)
    customer_df["Mobile"] = customer_df["Mobile"].str.slice_replace(stop=1, repl='966')

    out = customer_df.drop_duplicates(subset=cols)

    out = out.astype(str)

    return out


# Make-column normalisation rules. Single regex pass instead of dozens of
# chained .str.replace() calls. Keys are matched as literal substrings
# (regex=False semantics) via Series.replace with a compiled pattern map.
_MAKE_REPLACEMENTS = {
    "-": " ",
    ".": "",
    "cherry": "chery",
    "chevorlet": "chevrolet",
    "chevrolete": "chevrolet",
    "cheverolet": "chevrolet",
    "dihatsu": "daihatsu",
    "daihatzu": "daihatsu",
    "emegrand": "emgrand",
    "great wall": "gwm",
    "gelly": "geely",
    "hino 300": "hino",
    "300 hino": "hino",
    "hyudai": "hyundai",
    "hyundia": "hyundai",
    "hundai": "hyundai",
    "izusu": "isuzu",
    "izuzu": "isuzu",
    "infinitiy nissan": "infiniti",
    "infinity nissan": "infiniti",
    "range rover": "land rover",
    "masda": "mazda",
    "mazda6": "mazda",
    "mazda 6": "mazda",
    "mercedez": "mercedes",
    "mitshubishi": "mitsubishi",
    "mitsubushi": "mitsubishi",
    "mitzubishi": "mitsubishi",
    "mitsubitshi": "mitsubishi",
    "mitsubitsi": "mitsubishi",
    "pajero": "mitsubishi",
    "nisan": "nissan",
    "nissan diesel": "nissan",
    "peugeut": "peugeot",
    "duster": "renault",
    "renult": "renault",
    "renualt": "renault",
    "ZUSUKI": "suzuki",
    "suzuki dzire": "suzuki",
    "camry": "toyota",
    "toyata": "toyota",
    "toyoya": "toyota",
    "zxauto": "zx auto",
    "benz": "",
    "bens": "",
    "(china)": "",
    "trucks": "",
}

# Model-column normalisation rules (all the chained .str.replace blocks merged).
# Order is preserved since dicts are ordered in modern Python, matching the
# left-to-right semantics of the original chain.
_MODEL_REPLACEMENTS = {
    "-": " ",
    ".": "",
    "mazda3": "3",
    "mazda 3": "3",
    "mg 5": "5",
    "mg 6": "6",
    "6 (gl)": "6",
    "mazda6": "6",
    "mazda 6": "6",
    "emgrand7": "7",
    "emgrand8": "8",
    "accord + coup? v": "accord",
    "cr v": "crv",
    "camry (asv50)": "camry",
    "camry (axvh71)": "camry",
    "carry": "camry",
    "corola": "corolla",
    "corolla im": "corolla",
    "corolla (zre171)": "corolla",
    "corrola": "corolla",
    "corrolla": "corolla",
    "mazda cx 30": "cx 30",
    "mazda cx 5": "cx 5",
    "mazda cx 9": "cx 9",
    "cx 9 (tc)": "cx 9",
    "cx3": "cx 3",
    "cx30": "cx 30",
    "cx5": "cx 5",
    "cx9": "cx 9",
    "d max (sa)": "dmax",
    "d max": "dmax",
    "elantra coupe": "elantra",
    "elantra gt": "elantra",
    "elantra (g4n)": "elantra",
    "elantra (g4f)": "elantra",
    "elentra": "elantra",
    "elantra1": "elantra",
    "es 350": "es350",
    "es 300": "es300",
    "expedition el": "expedition",
    "expidetion": "expedition",
    "expedetion": "expedition",
    "expedation": "expedition",
    "explorer i": "explorer",
    "expidition": "expedition",
    "escalade esv": "escalade",
    "f 150": "f150",
    "f150 pickup": "f150",
    "fotuner": "fortuner",
    "fortuner ggn 155 &165": "fortuner",
    "fortuner (ggn155,ggn165)": "fortuner",
    "fortuner (sa)": "fortuner",
    "fortuner (tgn156,tgn166)": "fortuner",
    "gs 350": "gs350",
    "gs 430": "gs430",
    "h 1": "h1",
    "hi ace": "hiace",
    "hiace (sa)": "hiace",
    "hiace (trh201)": "hiace",
    "hiace van": "hiace",
    "hiace trh 201": "hiace",
    "hi lux": "hilux",
    "hillux": "hilux",
    "hilux (sa)": "hilux",
    "hilux (tgn111)": "hilux",
    "hilux (tgn121)": "hilux",
    "hilux (tgn126)": "hilux",
    "hilux (trh201)": "hilux",
    "l 200": "l200",
    "triton (l200)": "l200",
    "landcruiser": "land cruiser",
    "land cruiser (urj202)": "land cruiser",
    "land crusier": "land cruiser",
    "land cruiser (urj200)": "land cruiser",
    "land cruiser (sa)": "land cruiser",
    "land cruiser / land cruiser prado": "land cruiser prado",
    "land cruiser / prado": "land cruiser prado",
    "ls 400": "ls400",
    "ls 430": "ls430",
    "lx 470": "lx470",
    "lx 570": "lx570",
    "ls 460": "ls460",
    "navara (4x4)": "navara",
    "navara (d23)": "navara",
    "navara (d40)": "navara",
    "navarra": "navara",
    "patrol(y62)": "patrol",
    "patrol (y61)": "patrol",
    "patrol (y62)": "patrol",
    "patrol (y62) (vk56de)": "patrol",
    "patrol (vk56de)": "patrol",
    "patrol (new)": "patrol",
    "patrol new": "patrol",
    "patrol gr ii": "patrol",
    "patrol safari": "patrol",
    "patrol pickup (sa)": "patrol",
    "patrol i": "patrol",
    "patrol ii": "patrol",
    "patrol 4x4": "patrol",
    "patrol gr (sa)": "patrol",
    "patrol suv (sa)": "patrol",
    "patrol platinum": "patrol",
    "patroli": "patrol",
    "nissan patrol": "patrol",
    "pic up": "pick up",
    "rav 4": "rav4",
    "santafe": "santa fe",
    "santa fe xl": "santa fe",
    "sierra 1500 pickup": "sierra 1500",
    "sierra 1500 hd": "sierra 1500",
    "sierra 2500 pickup": "sierra 2500",
    "sierra 2500 hd": "sierra 2500",
    "silverado 1500 pickup": "silverado 1500",
    "sunny (b15)": "sunny",
    "sunny (n17)": "sunny",
    "taunus": "taurus",
    "taurus x": "taurus",
    "tauros": "taurus",
    "trail blazer": "trailblazer",
    "trailblazer ext": "trailblazer",
    "x trail (t31)": "xtrail",
    "x trail (t32)": "xtrail",
    "x trail": "xtrail",
    "yaris i / yaris verso (p1)": "yaris",
    "yaris ia": "yaris",
    "yaris (ncp151)": "yaris",
    "yaris & yaris sedan": "yaris",
    "yariz": "yaris",
    "yukon denali xl": "yukon denali",
    "yukon xl denali": "yukon denali",
    "yukon 1500": "yukon",
    "yukon xl 1500": "yukon xl",
    "yukon xl 2500": "yukon xl",
    "yukonxl": "yukon xl",
    "yukon yukon": "yukon",
    "mg zs": "zs",
    "zst": "zs",
}


def _apply_literal_replacements(series, mapping):
    """Apply a mapping of literal substring replacements in order.

    Faster than chaining N str.replace calls because each chained call
    rebuilds the Series; this stays in a single object and keeps each
    individual replace cheap (no regex compilation for literal strings).
    """
    for old, new in mapping.items():
        series = series.str.replace(old, new, regex=False)
    return series


def ingest_vehicles():
    with _db_conn() as conn:
        vehicles_PE_df = _get_all_data_from_table(conn, "v_Vehicle")
        vehicles_PAC_df = _get_all_data_from_table(conn, "v_PAC_Vehicle")

    cols = vehicles_PE_df.columns

    vehicles_PE_df["StationBrand"] = "PE"
    vehicles_PAC_df["StationBrand"] = "PAC"

    vehicles_df = pd.concat([vehicles_PE_df, vehicles_PAC_df], axis=0).drop_duplicates(subset=cols)

    logger.info("adjust Maker")

    vehicles_df["Make"] = vehicles_df["Make"].str.lower()
    vehicles_df["is_truck"] = vehicles_df["Make"].str.contains("truck").fillna(False).astype("int")
    vehicles_df["Make"] = _apply_literal_replacements(vehicles_df["Make"], _MAKE_REPLACEMENTS).str.strip()

    logger.info("adjust Model")

    vehicles_df["Model"] = vehicles_df["Model"].str.lower()
    vehicles_df["Model"] = _apply_literal_replacements(vehicles_df["Model"], _MODEL_REPLACEMENTS)
    # The original code also had a single regex replacement for ^denali$ -> yukon denali,
    # which can't be expressed as a literal substring rule.
    vehicles_df["Model"] = vehicles_df["Model"].str.replace(r"^denali$", "yukon denali", regex=True).str.strip()

    logger.info("adjust price level")

    # NOTE: this dict has duplicate keys in the original code, so the later
    # value silently overrides the earlier one. Behaviour preserved here:
    #   "chevrolet" -> "low" (overrides "medium")
    #   "ford"      -> "medium" (overrides "low")
    #   "lincoln"   -> "high"
    #   "mercedez"  -> "very_high"  (probable typo for "mercedes")
    # Worth reviewing whether that's intended.
    maker_map = {
        # Very High
        "porsche": "very_high",
        "lamborghini": "very_high",
        "maserati": "very_high",
        "bmw": "very_high",
        "volvo": "very_high",
        "jaguar": "very_high",
        "mercedez": "very_high",
        "chrysler": "very_high",
        "dodge": "very_high",
        "audi": "very_high",
        # High
        "toyota": "high",
        "byd": "high",
        "honda": "high",
        "lexus": "high",
        "jeep": "high",
        "gmc": "high",
        "lincoln": "high",
        "chevrolet trucks": "high",
        "ford trucks": "high",
        "gmc trucks": "high",
        "lincoln": "high",
        # Medium
        "volkswagen": "medium",
        "chevrolet": "medium",
        "fiat": "medium",
        "ford": "medium",
        "nissan": "medium",
        "mitsubishi": "medium",
        "ford": "medium",
        "mercury": "medium",
        # Low
        "pegout": "low",
        "renault": "low",
        "suzuki": "low",
        "chevrolet": "low",
        "chevrolet": "low",
        # Very Low
        "jac": "very_low",
        "chery": "very_low",
    }

    vehicles_df["vehicle_brand_level"] = vehicles_df["Make"].map(maker_map).fillna("other")

    out = vehicles_df.drop_duplicates(subset=cols)

    out["PlateNumber"] = out["PlateNumber"].str.zfill(7)

    out = out.astype(str)

    return out


def ingest_branches():
    with _db_conn() as conn:
        branches_PE_df = _get_all_data_from_table(conn, "v_Branch")
        branches_PAC_df = _get_all_data_from_table(conn, "v_PAC_Branch")

    cols = branches_PE_df.columns

    branches_PE_df["StationBrand"] = "PE"
    branches_PAC_df["StationBrand"] = "PAC"

    branches_df = pd.concat([branches_PE_df, branches_PAC_df], axis=0).drop_duplicates(subset=cols)

    # Swap lat/lon when they're flipped (lat > lon shouldn't happen at this site's coords).
    # Vectorised swap without the temporary column.
    flipped = branches_df["Latitude"] > branches_df["Longitude"]
    lat = branches_df["Latitude"].copy()
    branches_df.loc[flipped, "Latitude"] = branches_df.loc[flipped, "Longitude"]
    branches_df.loc[flipped, "Longitude"] = lat[flipped]

    out = branches_df.drop_duplicates(subset=cols)

    out = out.astype(str)

    return out


def ingest_invoices():
    logger.info("start downloading invoices")

    with _db_conn() as conn:
        invoices_PE_df = get_all_data_from_invoice_table(conn, "v_Invoice")
        invoices_PAC_df = get_all_data_from_invoice_table(conn, "v_PAC_Invoice")

    cols = invoices_PE_df.columns

    invoices_PE_df["StationBrand"] = "PE"
    invoices_PAC_df["StationBrand"] = "PAC"

    logger.info("concat invoices")

    invoice_df = pd.concat([invoices_PE_df, invoices_PAC_df], axis=0).drop_duplicates(subset=cols)

    invoice_df["InvoiceID"] = invoice_df["InvoiceID"].astype("string")

    invoice_df = invoice_df.astype(str)

    return invoice_df


def ingest_invoices_items_PE(filepath, min_date):
    logger.info("start downloading invoices items")

    with _db_conn() as conn:
        invoicesitems_PE_df = get_all_data_from_invoice_items_table(conn, "v_Invoice", "v_InvoiceItems", filepath, min_date)

    logger.info("Drop duplicates")

    cols = invoicesitems_PE_df.columns

    out = invoicesitems_PE_df.drop_duplicates(subset=cols)

    del invoicesitems_PE_df

    logger.info("Config cols")

    out = out.astype(str)

    return out


def ingest_invoices_items_PAC(filepath, min_date):
    logger.info("start downloading invoices items")

    with _db_conn() as conn:
        invoicesitems_PAC_df = get_all_data_from_invoice_items_table(conn, "v_PAC_Invoice", "v_PAC_InvoiceItems", filepath, min_date)

    logger.info("Drop duplicates")

    cols = invoicesitems_PAC_df.columns

    out = invoicesitems_PAC_df.drop_duplicates(subset=cols)

    del invoicesitems_PAC_df

    logger.info("Config cols")

    out = out.astype(str)

    return out


def ingest_invoices_items(invoicesitems_PE_df, invoicesitems_PAC_df):

    cols = invoicesitems_PE_df.columns

    invoicesitems_PE_df = invoicesitems_PE_df.astype(str)
    invoicesitems_PAC_df = invoicesitems_PAC_df.astype(str)

    invoicesitems_PE_df["StationBrand"] = "PE"
    invoicesitems_PAC_df["StationBrand"] = "PAC"

    logger.info("concat invoices items")

    invoice_items_df = pd.concat([invoicesitems_PE_df, invoicesitems_PAC_df], axis=0).drop_duplicates(subset=cols)

    logger.info("delete invoices items")
    del invoicesitems_PE_df
    del invoicesitems_PAC_df

    invoice_items_df["InvoiceID"] = invoice_items_df["InvoiceID"].astype("string")

    float_cols = [
        "ServiceTotalAmount", "ItemTotalAmount", "ServiceBeforeTaxAmount",
        "ItemBeforeTaxAmount", "ServiceBeforeDiscountAmount",
        "ItemBeforeDiscountAmount", "ServiceTotalDiscountAmount",
        "ItemTotalDiscountAmount", "ServiceItemCostAmount",
    ]
    invoice_items_df[float_cols] = invoice_items_df[float_cols].astype("float64")

    # Compute group-default masks once instead of recomputing isnull() five times
    service_null = invoice_items_df["ServiceItemGroupDefaultName"].isnull()
    invoice_items_df["ItemBaseQuantity"] = np.where(service_null, 1, invoice_items_df["ItemBaseQuantity"])
    invoice_items_df.loc[service_null, "ServiceItemDefaultName"] = "Service"
    invoice_items_df.loc[service_null, "ServiceItemCode"] = "Service"
    invoice_items_df.loc[service_null, "ServiceItemGroupDefaultName"] = "Service"

    invoice_items_df["sku"] = (
        invoice_items_df["ServiceName"] + " | "
        + invoice_items_df["ServiceItemGroupDefaultName"] + " | "
        + invoice_items_df["ServiceItemDefaultName"] + " | "
        + invoice_items_df["ServiceItemCode"] + " | "
        + invoice_items_df["ServicePackageName"]
    )
    invoice_items_df["InvoiceTotalAmount"] = invoice_items_df["ServiceTotalAmount"] + invoice_items_df["ItemTotalAmount"]
    invoice_items_df["InvoiceBeforeTaxAmount"] = invoice_items_df["ServiceBeforeTaxAmount"] + invoice_items_df["ItemBeforeTaxAmount"]
    invoice_items_df["InvoiceBeforeDiscountAmount"] = invoice_items_df["ServiceBeforeDiscountAmount"] + invoice_items_df["ItemBeforeDiscountAmount"]
    invoice_items_df["InvoiceTotalDiscountAmount"] = invoice_items_df["ServiceTotalDiscountAmount"] + invoice_items_df["ItemTotalDiscountAmount"]
    invoice_items_df["InvoiceGrossMargin"] = invoice_items_df["InvoiceBeforeTaxAmount"] - invoice_items_df["ServiceItemCostAmount"]

    invoice_items_df = invoice_items_df.astype(str)

    return invoice_items_df


def ingest_transactions(invoice_df, invoice_items_df):

    invoice_cols = ["InvoiceID", "CustomerID", "CustomerVehicleID", "BranchID", "InvoiceDate", "IsFleet", "IsPMS", "WorkOrderMileage", "PreviousMileage", "NewCustomer", "NewVehicle", "Warranty"]
    invoice_items_cols = ["InvoiceID", "ServiceName", "ServiceItemGroupDefaultName", "ServiceItemDefaultName", "ServiceItemCode", "ServicePackageName", "sku", "ItemBaseQuantity", "ServiceItemCostAmount", "InvoiceTotalDiscountAmount", "InvoiceBeforeDiscountAmount", 'InvoiceBeforeTaxAmount', 'InvoiceTotalAmount', "InvoiceGrossMargin"]

    logger.info("Merge invoice and invoice items")

    transactions_df = pd.merge(
        invoice_df[invoice_cols],
        invoice_items_df[invoice_items_cols],
        on="InvoiceID",
        how="inner"
    )

    logger.info("delete invoices df")
    del invoice_items_df
    del invoice_df

    logger.info("GroupBy transactions")

    float_cols = [
        "IsFleet", "IsPMS", "WorkOrderMileage", "PreviousMileage",
        "NewCustomer", "NewVehicle", "Warranty", "ItemBaseQuantity",
        "ServiceItemCostAmount", "InvoiceTotalDiscountAmount",
        "InvoiceBeforeTaxAmount", "InvoiceBeforeDiscountAmount",
        "InvoiceTotalAmount", "InvoiceGrossMargin",
    ]
    transactions_df[float_cols] = transactions_df[float_cols].astype("float64")

    transactions_grouped_df = transactions_df.groupby(
        ["InvoiceID", "CustomerID", "CustomerVehicleID", "BranchID", "InvoiceDate", "ServiceName", "ServiceItemGroupDefaultName", "ServiceItemDefaultName", "ServiceItemCode", "ServicePackageName", "sku"],
        as_index=False
    ).agg({
        "IsFleet": "max",
        "IsPMS": "max",
        "WorkOrderMileage": "max",
        "PreviousMileage": "max",
        "NewCustomer": "max",
        "NewVehicle": "max",
        "Warranty": "max",
        "ItemBaseQuantity": "sum",
        "ServiceItemCostAmount": "sum",
        "InvoiceTotalDiscountAmount": "sum",
        "InvoiceBeforeTaxAmount": "sum",
        "InvoiceBeforeDiscountAmount": "sum",
        "InvoiceTotalAmount": "sum",
        "InvoiceGrossMargin": "sum",
    })

    logger.info("Final Features")

    transactions_grouped_df["InvoiceGrossMargin_perc"] = (
        transactions_grouped_df["InvoiceGrossMargin"] / transactions_grouped_df["InvoiceTotalAmount"]
    ).fillna(0)
    transactions_grouped_df["hasDiscount"] = (transactions_grouped_df["InvoiceTotalDiscountAmount"] > 0).astype(int)

    transactions_grouped_df["MileageBetweenVisits"] = transactions_grouped_df["WorkOrderMileage"] - transactions_grouped_df["PreviousMileage"]
    transactions_grouped_df["MileageBetweenVisits_perc"] = transactions_grouped_df["MileageBetweenVisits"] / transactions_grouped_df["PreviousMileage"]

    logger.info("Save")

    out = transactions_grouped_df.astype(str)

    return out


# 12 minutes
def ingestion_general():
    logger.info("Branches")
    out = ingest_branches()
    out.to_parquet("data/01_raw/raw_branches.parquet", index=False)

    # logger.info("Promo")
    # out = ingest_promos()
    # out.to_parquet("data/01_raw/raw_promos.parquet", index=False)

    logger.info("Customers")
    out = ingest_customers()
    out.to_parquet("data/01_raw/raw_customers.parquet", index=False)

    logger.info("Vehicles")
    out = ingest_vehicles()
    out.to_parquet("data/01_raw/raw_vehicles.parquet", index=False)


# 20 minutes
def ingestion_invoices():
    logger.info("Invoices")
    invoice_df = ingest_invoices()
    invoice_df.to_parquet("data/01_raw/raw_invoices.parquet", index=False)


# 25 minutes
def ingestion_invoice_items_PE(min_date="2025-08-01"):
    logger.info("Items PE")
    filepath = "data/01_raw/raw_invoices_items_PE_files"
    invoicesitems_PE_df = ingest_invoices_items_PE(filepath, min_date)
    logger.info("Saving Items PE")
    invoicesitems_PE_df.to_parquet("data/01_raw/raw_invoices_items_PE.parquet", index=False, engine="fastparquet")


def ingestion_invoice_items_PAC(min_date="2025-08-01"):
    logger.info("Items PAC")
    filepath = "data/01_raw/raw_invoices_items_PAC_files"
    invoicesitems_PAC_df = ingest_invoices_items_PAC(filepath, min_date)
    logger.info("Saving Items PAC")
    invoicesitems_PAC_df.to_parquet("data/01_raw/raw_invoices_items_PAC.parquet", index=False)


def ingestion_items():
    logger.info("Read Files")
    invoicesitems_PE_df = pd.read_parquet("data/01_raw/raw_invoices_items_PE.parquet",)
    invoicesitems_PAC_df = pd.read_parquet("data/01_raw/raw_invoices_items_PAC.parquet",)

    logger.info("Items")
    invoice_items_df = ingest_invoices_items(
        invoicesitems_PE_df,
        invoicesitems_PAC_df,
    )

    logger.info("Delete Files")
    del invoicesitems_PE_df
    del invoicesitems_PAC_df

    logger.info("Writing Files")
    invoice_items_df.to_parquet("data/01_raw/raw_invoices_items.parquet", index=False)


def prepare_transactions():
    logger.info("Read Files")
    invoice_df = pd.read_parquet("data/01_raw/raw_invoices.parquet")
    invoice_items_df = pd.read_parquet("data/01_raw/raw_invoices_items.parquet")

    logger.info("Transactions")
    transactions_df = ingest_transactions(
        invoice_df,
        invoice_items_df,
    )

    logger.info("Delete Files")
    del invoice_items_df
    del invoice_df

    logger.info("Writing Files")
    transactions_df.to_parquet("data/01_raw/transactions_origin.parquet", index=False)


def main():

    min_date = "2026-05-01"

    logger.info("Start Ingestion Process")

    logger.info("Start General")
    ingestion_general()

    # # 20 minutes
    logger.info("Start Invoices")
    ingestion_invoices()

    # # 23 minutes
    logger.info("Start Items PE")
    ingestion_invoice_items_PE(min_date)

    # # 1 minute
    logger.info("Start Items PAC")
    ingestion_invoice_items_PAC(min_date)

    # #
    logger.info("Start Items")
    ingestion_items()

    # 20 minutos
    logger.info("Start Transactions")
    prepare_transactions()


if __name__ == "__main__":
        main()