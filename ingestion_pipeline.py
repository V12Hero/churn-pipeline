
import datetime
import pandas as pd
import numpy as np
import pyodbc

import os
import glob
import tqdm

import logging
from rich.logging import RichHandler

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


def _get_conn():
    """
    The function `_get_conn` establishes a connection to a SQL Server using the provided server,
    database, username, and password.

    Returns:
      The function `_get_conn` returns a connection object to a SQL Server database using the provided
    connection string.
    """

    connectionString = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};TrustServerCertificate=yes;'

    conn = pyodbc.connect(connectionString)
    return conn


def _get_all_data_from_table(conn, tablename):

    cursor = conn.cursor()

    cursor.execute(f"SELECT TOP 1 * FROM MACDB.dbo.{tablename}")
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]

    result_df = None

    if "ModifiedOn" in cols:
        cursor.execute(f"""
            SELECT 
                MIN(ModifiedOn)
            FROM MACDB.dbo.{tablename}
        """)
        rows = cursor.fetchall()

        min_date = rows[0][0]
        max_date = datetime.date.today()

        dates_list = pd.date_range(min_date, max_date, freq='1ME')

        tables_result_list = []
        for date in tqdm.tqdm(dates_list):
            cursor.execute(f"""
                SELECT 
                    *
                FROM MACDB.dbo.{tablename}
                WHERE YEAR(ModifiedOn) = {date.year} and Month(ModifiedOn) = {date.month}
            """)
            rows = cursor.fetchall()

            df = pd.DataFrame.from_records(rows, columns=[col[0] for col in cursor.description])
            if df.shape[0] > 0:
                tables_result_list.append(df)

        result_df = pd.concat(tables_result_list).reset_index(drop=True)

    else:
        cursor.execute(f"""
            SELECT 
                *
            FROM MACDB.dbo.{tablename}
        """)
        rows = cursor.fetchall()

        df = pd.DataFrame.from_records(rows, columns=[col[0] for col in cursor.description])

        result_df = df

    if result_df is None:
         raise ValueError("result_df is None. There is a problem with cursor")

    return result_df


def get_all_data_from_invoice_table(conn, tablename):

    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT 
            MIN(ModifiedOn)
        FROM MACDB.dbo.{tablename}
    """)
    rows = cursor.fetchall()

    min_date = rows[0][0]
    max_date = datetime.date.today() + datetime.timedelta(days=31)

    dates_list = pd.date_range(min_date, max_date, freq='1ME')

    print(dates_list.min(), " - ", dates_list.max())

    monthly_df_list = []

    for date in tqdm.tqdm(dates_list):
        cursor.execute(f"""
            SELECT 
                *
            FROM MACDB.dbo.{tablename}
            WHERE YEAR(ModifiedOn) = {date.year} and Month(ModifiedOn) = {date.month}
        """)
        rows = cursor.fetchall()

        invoice_df = pd.DataFrame.from_records(rows, columns=[col[0] for col in cursor.description])
        if invoice_df.shape[0] > 0:
            monthly_df_list.append(invoice_df)
            # invoice_df.to_parquet(f"{tablename}/{date.year}{date.month:02}.parquet")

    out = pd.concat(monthly_df_list, axis=0)

    return out


def get_all_data_from_invoice_items_table(conn, tablename, table_items_name, filepath, min_date):

    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT 
            MIN(ModifiedOn)
        FROM MACDB.dbo.{tablename}
    """)
    rows = cursor.fetchall()

    # min_date = rows[0][0]
    max_date = datetime.date.today() + datetime.timedelta(days=31)

    dates_list = pd.date_range(min_date, max_date, freq='1ME')
    print(dates_list.min(), " - ", dates_list.max())

    monthly_df_list = []

    try:
        os.mkdir(filepath)
        print(f"Directory '{filepath}' created successfully.")
    except FileExistsError:
        print(f"Directory '{filepath}' already exists.")

    for date in tqdm.tqdm(dates_list):
        cursor.execute(f"""
            SELECT 
                a.*
            FROM
                MACDB.dbo.{table_items_name} a
            inner join
                (SELECT InvoiceID, ModifiedOn FROM MACDB.dbo.{tablename}) b
            ON a.InvoiceID = b.InvoiceID
            WHERE YEAR(b.ModifiedOn) = {date.year} and Month(b.ModifiedOn) = {date.month}
        """)
        rows = cursor.fetchall()

        items_df = pd.DataFrame.from_records(rows, columns=[col[0] for col in cursor.description])
        if items_df.shape[0] > 0:
            # monthly_df_list.append(items_df)
            items_df.to_parquet(f"{filepath}/{date.year}{date.month:02}.parquet", index=False)

    # out = pd.concat(monthly_df_list, axis=0)
    out = pd.read_parquet(filepath)

    return out


def ingest_promos():
    conn = _get_conn()
    promos_df = _get_all_data_from_table(conn, "TMP_PROMOS")

    out = promos_df

    return out


def ingest_customers():
    conn = _get_conn()
    customer_PE_df = _get_all_data_from_table(conn, "v_Customer")
    customer_PAC_df = _get_all_data_from_table(conn, "v_PAC_Customer")

    cols = customer_PE_df.columns

    customer_PE_df["StationBrand"] = "PE"
    customer_PAC_df["StationBrand"] = "PAC"

    customer_df = pd.concat([customer_PE_df, customer_PAC_df], axis=0).drop_duplicates(subset=cols)

    customer_df["Mobile"] = customer_df["Mobile"].str.zfill(10)
    customer_df["Mobile"] = customer_df["Mobile"].str.slice_replace(stop=1, repl='966')

    out = customer_df.drop_duplicates(subset=cols)

    for col in out.columns:
        out[col] = out[col].astype("str")

    return out


# Make typos seen in MACDB (Nov 2026 audit of 7.8M vehicles). Exact match after the replace chain in
# ingest_vehicles(); add new misspellings under the right brand.
_MAKE_TYPOS = {
    "baic": [
        "baic d50", "baic x35",
    ],
    "bestune": [
        "benteng", "besturn",
    ],
    "cadillac": [
        "cadilac",
    ],
    "changan": [
        "chagan", "changa", "changgan", "changhan", "shangan",
    ],
    "chery": [
        "cheery", "chery tiggo",
    ],
    "chevrolet": [
        "chebrolet", "cherolet", "chervolet", "chevolet", "chevrolate", "chevrole", "chevroleit",
        "chevrolet grove", "chevrolet tahoe", "chevrolete", "chevrolett", "chevrolette", "chevroley",
        "chevrolit", "chevrolite", "chevrollet", "chevrollete", "chevy",
    ],
    "chrysler": [
        "crysler",
    ],
    "daihatsu": [
        "dahaitsu", "dahatsu", "dai hatsu", "daihasu", "daihatshu", "daihatso", "daihatsun", "daihitsu",
        "daihtsu", "daihutsu", "daithatsu", "daitshu", "daitsu", "dhaihatsu", "dhiatsu", "diahatsu",
        "diatshu", "diatsu", "dihatso", "dihatzu", "dihutsu",
    ],
    "dodge": [
        "dodge charger", "dodges", "dogde", "doge", "dogge", "doodge", "dudge",
    ],
    "ford": [
        "ford explorer",
    ],
    "geely": [
        "emgrand geely", "geeely", "geel", "geeley", "geelly", "geely emgrand", "geely motors", "gely",
        "gleely", "gleey", "greely", "jeely",
    ],
    "genesis": [
        "geneses", "genisis",
    ],
    "gmc": [
        "gm",
    ],
    "gwm": [
        "wingle",
    ],
    "haval": [
        "15 haval", "h6 haval", "h9 haval", "h9haval", "haval h2", "haval h6",
    ],
    "honda": [
        "handa", "honda 1",
    ],
    "hummer": [
        "hammer",
    ],
    "hyundai": [
        "hayundai", "huyndai", "hyndai", "hyuandai", "hyuidai", "hyuindai", "hyumdai", "hyunadi", "hyunda",
        "hyundae", "hyunday", "hyundi", "nyundai", "yhyundai",
    ],
    "infiniti": [
        "ifiniti nissan", "infiniti  nissan", "infiniti / nissan", "infiniti nissan", "infiniti/nissan",
        "infinitinissan", "infinity", "infinity/nissan", "infinti", "infinti nissan", "inifiniti nissan",
        "nissan infiniti", "nissan infinity", "nissan/infiniti", "nissan/infinity", "nissaninfiniti",
    ],
    "isuzu": [
        "1suzu", "asuzu", "isizu", "isusu", "isuz", "isuza", "isuzo", "isuzu d max", "isuzu d max 25td",
        "isuzu dmax", "iszu",
    ],
    "jac": [
        "jack",
    ],
    "jetour": [
        "15jetour", "jetour x70", "jetour x70 240t",
    ],
    "land rover": [
        "landrover",
    ],
    "lexus": [
        "lexus lx570", "lexus sport", "lexux", "lexuz", "lixus", "luxus",
    ],
    "lincoln": [
        "lincolin", "lincon",
    ],
    "mazda": [
        "maxda", "mazada", "mazda 3", "mazda cx9", "mazda3", "mazda9",
    ],
    "mercedes": [
        "actros", "marcedes", "mecedez", "mercdes", "mercedenz", "mercedes bench", "mercedese", "merceds",
        "mercides", "merecedes", "mersedes", "merzedes",
    ],
    "mg": [
        "m g",
    ],
    "mitsubishi": [
        "metsubishi", "mishubishi", "misthbishi", "mistsubishi", "mistubishi", "misubishi", "mitchubishi",
        "mithsubishi", "mitshibishi", "mitshubishe", "mitshubisi", "mitshubitshi", "mitsibishi",
        "mitsibushi", "mitsobishi", "mitsubeshi", "mitsubhisi", "mitsubihi", "mitsubihsi", "mitsubish",
        "mitsubishe", "mitsubishi fuso", "mitsubishi motors", "mitsubisi", "mitsubithi", "mitsubitzi",
        "mitsubshi", "mitsuibishi", "mitubishi", "mitusbishi", "mtsubishi", "mutsubishi",
    ],
    "nissan": [
        "i nissan", "inissan", "missan", "nassan", "nessan", "niisan", "niissan", "nis san", "nisaan",
        "nissa", "nissam", "nissan 1", "nissan datsun", "nissan patrol", "nissan pick up", "nissan sunny",
        "nissan*", "nissan,", "nissanh", "nissann", "nissans", "nissn", "nisssan", "nlssan", "nnissan",
        "nssan",
    ],
    "peugeot": [
        "pegeot", "peugot",
    ],
    "porsche": [
        "porche", "porshe",
    ],
    "renault": [
        "ranault", "reanault", "reanult", "renaul", "renaut", "renoult", "reynault",
    ],
    "subaru": [
        "subaro",
    ],
    "suzuki": [
        "susuki", "suzki", "zuzuki",
    ],
    "toyota": [
        "t0yota", "tayota", "totota", "totyota", "toy0ta", "toyoa", "toyot", "toyota corolla",
        "toyota fortuner", "toyota hilux", "toyota innova", "toyota land cruiser", "toyota prado",
        "toyota toyota", "toyota yaris", "toyotaa", "toyoto", "toyotya", "toyoyta", "toyta", "toytota",
        "toyuta", "tpyota", "tyota", "yoyota",
    ],
    "volkswagen": [
        "volkswagon", "volkwagen",
    ],
    "volvo": [
        "valvo", "velvo", "volov", "vovlo",
    ],
}
MAKE_FIXES = {typo: brand for brand, typos in _MAKE_TYPOS.items() for typo in typos}

# Model spellings that mean the same car. Exact match, applied after brackets and extra spaces are removed.
MODEL_FIXES = {
    "pickup": "pick up", "xterra": "x terra", "fjcruiser": "fj cruiser", "4runner": "4 runner",
    "prado": "land cruiser prado", "land cruiserprado": "land cruiser prado",
    "furtuner": "fortuner", "inova": "innova", "sequia": "sequoia", "sequioa": "sequoia",
    "sta fe": "santa fe", "san tafe": "santa fe", "sunny new": "sunny", "tida": "tiida", "civilan": "civilian",
    "serato": "cerato", "cerrato": "cerato", "ceratto": "cerato", "carnaval": "carnival", "carenz": "carens",
    "crownvictoria": "crown victoria", "eco sport": "ecosport", "grand marquiz": "grand marquis",
    "ford everest": "everest", "mu x": "mux", "m u x": "mux", "m ux": "mux", "d zire": "dzire",
    "mazda2": "2", "mazda 2": "2", "300 c": "300c", "l b": "lb", "rewardnp": "reward np",
}


def ingest_vehicles():
    conn = _get_conn()
    vehicles_PE_df = _get_all_data_from_table(conn, "v_Vehicle")
    vehicles_PAC_df = _get_all_data_from_table(conn, "v_PAC_Vehicle")

    cols = vehicles_PE_df.columns

    vehicles_PE_df["StationBrand"] = "PE"
    vehicles_PAC_df["StationBrand"] = "PAC"

    vehicles_df = pd.concat([vehicles_PE_df, vehicles_PAC_df], axis=0).drop_duplicates(subset=cols)

    logger.info("adjust Maker")

    vehicles_df["Make"] = vehicles_df["Make"].str.lower()
    vehicles_df["is_truck"] = vehicles_df["Make"].str.contains("truck").fillna(False).astype("int")
    vehicles_df["Make"] = (
        vehicles_df["Make"]
        .str.replace("-", " ")
        .str.replace(".", "")
        .str.replace("cherry", "chery")
        .str.replace("chevorlet", "chevrolet")
        .str.replace("chevrolete", "chevrolet")
        .str.replace("cheverolet", "chevrolet")
        .str.replace("dihatsu", "daihatsu")
        .str.replace("daihatzu", "daihatsu")
        .str.replace("emegrand", "emgrand")
        .str.replace("great wall", "gwm")
        .str.replace("gelly", "geely")
        .str.replace("hino 300", "hino")
        .str.replace("300 hino", "hino")
        .str.replace("hyudai", "hyundai")
        .str.replace("hyundia", "hyundai")
        .str.replace("hundai", "hyundai")
        .str.replace("izusu", "isuzu")
        .str.replace("izuzu", "isuzu")
        .str.replace("infinitiy nissan", "infiniti")
        .str.replace("infinity nissan", "infiniti")
        .str.replace("range rover", "land rover")
        .str.replace("masda", "mazda")
        .str.replace("mazda6", "mazda")
        .str.replace("mazda 6", "mazda")
        .str.replace("mercedez", "mercedes")
        .str.replace("mitshubishi", "mitsubishi")
        .str.replace("mitsubushi", "mitsubishi")
        .str.replace("mitzubishi", "mitsubishi")
        .str.replace("mitsubitshi", "mitsubishi")
        .str.replace("mitsubitsi", "mitsubishi")
        .str.replace("pajero", "mitsubishi")
        .str.replace("nisan", "nissan")
        .str.replace("nissan diesel", "nissan")
        .str.replace("peugeut", "peugeot")
        .str.replace("duster", "renault")
        .str.replace("renult", "renault")
        .str.replace("renualt", "renault")
        .str.replace("ZUSUKI", "suzuki")
        .str.replace("suzuki dzire", "suzuki")
        .str.replace("camry", "toyota")
        .str.replace("toyata", "toyota")
        .str.replace("toyoya", "toyota")
        .str.replace("zxauto", "zx auto")
        .str.replace("benz", "")
        .str.replace("bens", "")
        .str.replace("(china)", "")
        .str.replace("trucks", "")
        .str.rstrip()
        .str.lstrip()
    )
    vehicles_df["Make"] = vehicles_df["Make"].str.replace(r"\s+", " ", regex=True).str.strip().replace(MAKE_FIXES)

    logger.info("adjust Model")

    vehicles_df["Model"] = vehicles_df["Model"].str.lower()

    vehicles_df["Model"] = (
        vehicles_df["Model"]
        .str.replace("-", " ")
        .str.replace(".", "")
        .str.replace("mazda3", "3")
        .str.replace("mazda 3", "3")
        .str.replace("mg 5", "5")
        .str.replace("mg 6", "6")
        .str.replace("6 (gl)", "6")
        .str.replace("mazda6", "6")
        .str.replace("mazda 6", "6")
        .str.replace("emgrand7", "7")
        .str.replace("emgrand8", "8")
        .str.replace("accord + coup? v", "accord")
        .str.replace("cr v", "crv")
        .str.replace("camry (asv50)", "camry")
        .str.replace("camry (axvh71)", "camry")
        .str.replace("carry", "camry")
        .str.replace("corola", "corolla")
        .str.replace("corolla im", "corolla")
        .str.replace("corolla (zre171)", "corolla")
        .str.replace("corrola", "corolla")
        .str.replace("corrolla", "corolla")
        .str.replace("mazda cx 30", "cx 30")
        .str.replace("mazda cx 5", "cx 5")
        .str.replace("mazda cx 9", "cx 9")
        .str.replace("cx 9 (tc)", "cx 9")
        .str.replace("cx3", "cx 3")
        .str.replace("cx30", "cx 30")
        .str.replace("cx5", "cx 5")
        .str.replace("cx9", "cx 9")
        .str.replace("d max (sa)", "dmax")
        .str.replace("d max", "dmax")
        .str.replace("elantra coupe", "elantra")
        .str.replace("elantra gt", "elantra")
        .str.replace("elantra (g4n)", "elantra")
        .str.replace("elantra (g4f)", "elantra")
        .str.replace("elentra", "elantra")
        .str.replace("elantra1", "elantra")
        .str.replace("es 350", "es350")
        .str.replace("es 300", "es300")
        .str.replace("expedition el", "expedition")
        .str.replace("expidetion", "expedition")
        .str.replace("expedetion", "expedition")
        .str.replace("expedation", "expedition")
        .str.replace("explorer i", "explorer")
        .str.replace("expidition", "expedition")
        .str.replace("escalade esv", "escalade")
        .str.rstrip()
        .str.lstrip()
    )

    vehicles_df["Model"] = (
        vehicles_df["Model"]
        .str.replace("f 150", "f150")
        .str.replace("f150 pickup", "f150")
        .str.replace("fotuner", "fortuner")
        .str.replace("fortuner ggn 155 &165", "fortuner")
        .str.replace("fortuner (ggn155,ggn165)", "fortuner")
        .str.replace("fortuner (sa)", "fortuner")
        .str.replace("fortuner (tgn156,tgn166)", "fortuner")
        .str.replace("gs 350", "gs350")
        .str.replace("gs 430", "gs430")
        .str.replace("h 1", "h1")
        .str.replace("hi ace", "hiace")
        .str.replace("hiace (sa)", "hiace")
        .str.replace("hiace (trh201)", "hiace")
        .str.replace("hiace van", "hiace")
        .str.replace("hiace trh 201", "hiace")
        .str.replace("hi lux", "hilux")
        .str.replace("hillux", "hilux")
        .str.replace("hilux (sa)", "hilux")
        .str.replace("hilux (tgn111)", "hilux")
        .str.replace("hilux (tgn121)", "hilux")
        .str.replace("hilux (tgn126)", "hilux")
        .str.replace("hilux (trh201)", "hilux")
        .str.rstrip()
        .str.lstrip()
    )

    vehicles_df["Model"] = (
        vehicles_df["Model"]
        .str.replace("l 200", "l200")
        .str.replace("triton (l200)", "l200")
        .str.replace("landcruiser", "land cruiser")
        .str.replace("land cruiser (urj202)", "land cruiser")
        .str.replace("land crusier", "land cruiser")
        .str.replace("land cruiser (urj200)", "land cruiser")
        .str.replace("land cruiser (sa)", "land cruiser")
        .str.replace("land cruiser / land cruiser prado", "land cruiser prado")
        .str.replace("land cruiser / prado", "land cruiser prado")
        .str.replace("ls 400", "ls400")
        .str.replace("ls 430", "ls430")
        .str.replace("lx 470", "lx470")
        .str.replace("lx 570", "lx570")
        .str.replace("ls 460", "ls460")
        .str.replace("navara (4x4)", "navara")
        .str.replace("navara (d23)", "navara")
        .str.replace("navara (d40)", "navara")
        .str.replace("navarra", "navara")
        .str.rstrip()
        .str.lstrip()
    )

    vehicles_df["Model"] = (
        vehicles_df["Model"]
        .str.replace("patrol(y62)", "patrol")
        .str.replace("patrol (y61)", "patrol")
        .str.replace("patrol (y62)", "patrol")
        .str.replace("patrol (y62) (vk56de)", "patrol")
        .str.replace("patrol (vk56de)", "patrol")
        .str.replace("patrol (new)", "patrol")
        .str.replace("patrol new", "patrol")
        .str.replace("patrol gr ii", "patrol")
        .str.replace("patrol safari", "patrol")
        .str.replace("patrol pickup (sa)", "patrol")
        .str.replace("patrol i", "patrol")
        .str.replace("patrol ii", "patrol")
        .str.replace("patrol 4x4", "patrol")
        .str.replace("patrol gr (sa)", "patrol")
        .str.replace("patrol suv (sa)", "patrol")
        .str.replace("patrol platinum", "patrol")
        .str.replace("patroli", "patrol")
        .str.replace("nissan patrol", "patrol")
        .str.replace("pic up", "pick up")
        .str.replace("rav 4", "rav4")
        .str.replace("santafe", "santa fe")
        .str.replace("santa fe xl", "santa fe")
        .str.replace("sierra 1500 pickup", "sierra 1500")
        .str.replace("sierra 1500 hd", "sierra 1500")
        .str.replace("sierra 2500 pickup", "sierra 2500")
        .str.replace("sierra 2500 hd", "sierra 2500")
        .str.replace("silverado 1500 pickup", "silverado 1500")
        .str.replace("sunny (b15)", "sunny")
        .str.replace("sunny (n17)", "sunny")
        .str.replace("taunus", "taurus")
        .str.replace("taurus x", "taurus")
        .str.replace("tauros", "taurus")
        .str.replace("trail blazer", "trailblazer")
        .str.replace("trailblazer ext", "trailblazer")
        .str.replace("x trail (t31)", "xtrail")
        .str.replace("x trail (t32)", "xtrail")
        .str.replace("x trail", "xtrail")
        .str.replace("yaris i / yaris verso (p1)", "yaris")
        .str.replace("yaris ia", "yaris")
        .str.replace("yaris (ncp151)", "yaris")
        .str.replace("yaris & yaris sedan", "yaris")
        .str.replace("yariz", "yaris")
        .str.replace("^denali$", "yukon denali", regex=True)
        .str.replace("yukon denali xl", "yukon denali")
        .str.replace("yukon xl denali", "yukon denali")
        .str.replace("yukon 1500", "yukon")
        .str.replace("yukon xl 1500", "yukon xl")
        .str.replace("yukon xl 2500", "yukon xl")
        .str.replace("yukonxl", "yukon xl")
        .str.replace("yukon yukon", "yukon")
        .str.replace("mg zs", "zs")
        .str.replace("zst", "zs")
        .str.rstrip()
        .str.lstrip()
    )

    # generic clean-up (Nov 2026): engine/chassis codes in brackets, repeated spaces, spaced model numbers
    vehicles_df["Model"] = (
        vehicles_df["Model"]
        .str.replace(r"\(.*?\)", " ", regex=True)                                   # "tucson (g4n)" -> "tucson"
        .str.replace(r"\(.*$", " ", regex=True)                                      # unclosed bracket
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.replace(r"^land cruiser ?/ ?prado$", "land cruiser prado", regex=True)
        .str.replace(r"^(es|gs|is|ls|lx|rx|gx|nx) (\d{3})$", r"\1\2", regex=True)     # lexus "rx 350" -> "rx350"
        .str.replace(r"^([cesl]) (\d{3})$", r"\1\2", regex=True)                     # "c 200" -> "c200", "l 300" -> "l300"
        .str.replace(r"^i (\d{2})$", r"i\1", regex=True)                             # hyundai "i 10" -> "i10"
        .replace(MODEL_FIXES)
    )

    logger.info("adjust price level")

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

    vehicles_df["vehicle_brand_level"] = vehicles_df["Make"].map(maker_map)
    vehicles_df["vehicle_brand_level"] = vehicles_df["vehicle_brand_level"].replace(np.nan, "other")

    out = vehicles_df.drop_duplicates(subset=cols)

    out["PlateNumber"] = out["PlateNumber"].str.zfill(7)

    for col in out.columns:
        out[col] = out[col].astype("str")

    return out


def ingest_branches():
    conn = _get_conn()
    branches_PE_df = _get_all_data_from_table(conn, "v_Branch")
    branches_PAC_df = _get_all_data_from_table(conn, "v_PAC_Branch")

    cols = branches_PE_df.columns

    branches_PE_df["StationBrand"] = "PE"
    branches_PAC_df["StationBrand"] = "PAC"

    branches_df = pd.concat([branches_PE_df, branches_PAC_df], axis=0).drop_duplicates(subset=cols)

    branches_df["new_latitude"] = np.where(branches_df["Latitude"] > branches_df["Longitude"], branches_df["Longitude"], branches_df["Latitude"])
    branches_df["Longitude"] = np.where(branches_df["Latitude"] > branches_df["Longitude"], branches_df["Latitude"], branches_df["Longitude"])
    branches_df["Latitude"] = branches_df["new_latitude"].copy()
    branches_df = branches_df.drop(columns="new_latitude")

    out = branches_df.drop_duplicates(subset=cols)

    for col in out.columns:
        out[col] = out[col].astype("str")

    return out


def ingest_invoices():

    conn = _get_conn()

    logger.info("start downloading invoices")

    invoices_PE_df = get_all_data_from_invoice_table(conn, "v_Invoice")
    invoices_PAC_df = get_all_data_from_invoice_table(conn, "v_PAC_Invoice")

    cols = invoices_PE_df.columns

    invoices_PE_df["StationBrand"] = "PE"
    invoices_PAC_df["StationBrand"] = "PAC"

    logger.info("concat invoices")

    invoice_df = pd.concat([invoices_PE_df, invoices_PAC_df], axis=0).drop_duplicates(subset=cols)

    invoice_df["InvoiceID"] = invoice_df["InvoiceID"].astype("string")

    for col in invoice_df.columns:
        invoice_df[col] = invoice_df[col].astype("str")

    return invoice_df


def ingest_invoices_items_PE(filepath, min_date):
    conn = _get_conn()

    logger.info("start downloading invoices items")

    invoicesitems_PE_df = get_all_data_from_invoice_items_table(conn, "v_Invoice", "v_InvoiceItems", filepath, min_date)

    logger.info("Drop duplicates")

    cols = invoicesitems_PE_df.columns

    out = invoicesitems_PE_df.drop_duplicates(subset=cols)

    del invoicesitems_PE_df

    logger.info("Config cols")

    for col in out.columns:
        out[col] = out[col].astype("str")

    return out


def ingest_invoices_items_PAC(filepath, min_date):
    conn = _get_conn()

    logger.info("start downloading invoices items")

    invoicesitems_PAC_df = get_all_data_from_invoice_items_table(conn, "v_PAC_Invoice", "v_PAC_InvoiceItems", filepath, min_date)

    logger.info("Drop duplicates")

    cols = invoicesitems_PAC_df.columns

    out = invoicesitems_PAC_df.drop_duplicates(subset=cols)

    del invoicesitems_PAC_df

    logger.info("Config cols")

    for col in out.columns:
        out[col] = out[col].astype("str")

    return out


def ingest_invoices_items(invoicesitems_PE_df, invoicesitems_PAC_df):

    cols = invoicesitems_PE_df.columns

    for col in invoicesitems_PE_df.columns:
        invoicesitems_PE_df[col] = invoicesitems_PE_df[col].astype("str")

    for col in invoicesitems_PAC_df.columns:
        invoicesitems_PAC_df[col] = invoicesitems_PAC_df[col].astype("str")

    invoicesitems_PE_df["StationBrand"] = "PE"
    invoicesitems_PAC_df["StationBrand"] = "PAC"

    logger.info("concat invoices items")

    invoice_items_df = pd.concat([invoicesitems_PE_df, invoicesitems_PAC_df], axis=0).drop_duplicates(subset=cols)

    logger.info("delete invoices items")
    del invoicesitems_PE_df
    del invoicesitems_PAC_df

    invoice_items_df["InvoiceID"] = invoice_items_df["InvoiceID"].astype("string")

    for col in ["ServiceTotalAmount", "ItemTotalAmount", "ServiceBeforeTaxAmount", "ItemBeforeTaxAmount", "ServiceBeforeDiscountAmount", "ItemBeforeDiscountAmount", "ServiceTotalDiscountAmount", "ItemTotalDiscountAmount", "ServiceItemCostAmount"]:
        invoice_items_df[col] = invoice_items_df[col].astype("float64")

    invoice_items_df["ItemBaseQuantity"] = np.where(invoice_items_df["ServiceItemGroupDefaultName"].isnull(), 1, invoice_items_df["ItemBaseQuantity"])
    invoice_items_df["ServiceItemDefaultName"] = np.where(invoice_items_df["ServiceItemGroupDefaultName"].isnull(), "Service", invoice_items_df["ServiceItemDefaultName"])
    invoice_items_df["ServiceItemGroupDefaultName"] = np.where(invoice_items_df["ServiceItemGroupDefaultName"].isnull(), "Service", invoice_items_df["ServiceItemGroupDefaultName"])
    invoice_items_df["ServiceItemCode"] = np.where(invoice_items_df["ServiceItemGroupDefaultName"].isnull(), "Service", invoice_items_df["ServiceItemCode"])
    invoice_items_df["sku"] = invoice_items_df["ServiceName"] + " | " +  invoice_items_df["ServiceItemGroupDefaultName"] + " | " + invoice_items_df["ServiceItemDefaultName"] + " | " + invoice_items_df["ServiceItemCode"] + " | " + invoice_items_df["ServicePackageName"]
    invoice_items_df["InvoiceTotalAmount"] = invoice_items_df["ServiceTotalAmount"] + invoice_items_df["ItemTotalAmount"]
    invoice_items_df["InvoiceBeforeTaxAmount"] = invoice_items_df["ServiceBeforeTaxAmount"] + invoice_items_df["ItemBeforeTaxAmount"]
    invoice_items_df["InvoiceBeforeDiscountAmount"] = invoice_items_df["ServiceBeforeDiscountAmount"] + invoice_items_df["ItemBeforeDiscountAmount"]
    invoice_items_df["InvoiceTotalDiscountAmount"] = invoice_items_df["ServiceTotalDiscountAmount"] + invoice_items_df["ItemTotalDiscountAmount"]
    invoice_items_df["InvoiceGrossMargin"] = invoice_items_df["InvoiceBeforeTaxAmount"] - invoice_items_df["ServiceItemCostAmount"]

    for col in invoice_items_df.columns:
        invoice_items_df[col] = invoice_items_df[col].astype("str")

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

    for col in ["IsFleet", "IsPMS", "WorkOrderMileage", "PreviousMileage", "NewCustomer", "NewVehicle", "Warranty", "ItemBaseQuantity", "ServiceItemCostAmount", "InvoiceTotalDiscountAmount", "InvoiceBeforeTaxAmount", "InvoiceBeforeDiscountAmount", "InvoiceTotalAmount", "InvoiceGrossMargin"]:
        transactions_df[col] = transactions_df[col].astype("float64")

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

    transactions_grouped_df["InvoiceGrossMargin_perc"] = transactions_grouped_df["InvoiceGrossMargin"] / transactions_grouped_df["InvoiceTotalAmount"]
    transactions_grouped_df["InvoiceGrossMargin_perc"] = np.where(transactions_grouped_df["InvoiceGrossMargin_perc"].isnull(), 0, transactions_grouped_df["InvoiceGrossMargin_perc"])
    transactions_grouped_df["hasDiscount"] = np.where(transactions_grouped_df["InvoiceTotalDiscountAmount"] > 0, 1, 0)

    transactions_grouped_df["MileageBetweenVisits"] = transactions_grouped_df["WorkOrderMileage"] - transactions_grouped_df["PreviousMileage"]
    transactions_grouped_df["MileageBetweenVisits_perc"] = transactions_grouped_df["MileageBetweenVisits"] / transactions_grouped_df["PreviousMileage"]

    logger.info("Save")

    out = transactions_grouped_df

    for col in out.columns:
        out[col] = out[col].astype("str")

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

    min_date = "2026-09-01"

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
