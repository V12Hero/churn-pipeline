import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import glob
import tqdm
import logging
from rich.logging import RichHandler

# --- CONFIGURATION ---
SERVER = '20.203.36.211'
DATABASE = 'MACDB'
USERNAME = 'to.mckinsey'
PASSWORD = 'Petromin@1'

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

# --- CONNECTION HELPER ---
def _get_engine():
    """
    Creates a SQLAlchemy engine using pymssql. 
    This is native to Mac and handles large data streams efficiently.
    """
    conn_url = f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    # use pool_pre_ping to ensure connection health during long ingestions
    return create_engine(conn_url, pool_pre_ping=True)

# --- EXTRACTION HELPERS ---
def _get_all_data_from_table(engine, tablename):
    with engine.connect() as conn:
        # Get one row to check for columns
        sample = pd.read_sql(text(f"SELECT TOP 1 * FROM MACDB.dbo.{tablename}"), conn)
        cols = sample.columns.tolist()

        if "ModifiedOn" in cols:
            min_date = conn.execute(text(f"SELECT MIN(ModifiedOn) FROM MACDB.dbo.{tablename}")).scalar()
            max_date = datetime.date.today()
            dates_list = pd.date_range(min_date, max_date, freq='1ME')

            tables_result_list = []
            for date in tqdm.tqdm(dates_list, desc=f"Loading {tablename}"):
                query = text(f"""
                    SELECT * FROM MACDB.dbo.{tablename}
                    WHERE YEAR(ModifiedOn) = {date.year} AND MONTH(ModifiedOn) = {date.month}
                """)
                df = pd.read_sql(query, conn)
                if not df.empty:
                    tables_result_list.append(df)
            
            if not tables_result_list:
                raise ValueError(f"No data found in {tablename}")
            
            result_df = pd.concat(tables_result_list).reset_index(drop=True)
        else:
            result_df = pd.read_sql(text(f"SELECT * FROM MACDB.dbo.{tablename}"), conn)

    return result_df

def get_all_data_from_invoice_table(engine, tablename):
    with engine.connect() as conn:
        min_date = conn.execute(text(f"SELECT MIN(ModifiedOn) FROM MACDB.dbo.{tablename}")).scalar()
        max_date = datetime.date.today() + datetime.timedelta(days=31)
        dates_list = pd.date_range(min_date, max_date, freq='1ME')

        print(f"Date Range: {dates_list.min()} - {dates_list.max()}")
        monthly_df_list = []

        for date in tqdm.tqdm(dates_list, desc=f"Invoices: {tablename}"):
            query = text(f"""
                SELECT * FROM MACDB.dbo.{tablename}
                WHERE YEAR(ModifiedOn) = {date.year} AND MONTH(ModifiedOn) = {date.month}
            """)
            df = pd.read_sql(query, conn)
            if not df.empty:
                monthly_df_list.append(df)

        return pd.concat(monthly_df_list, axis=0)

def get_all_data_from_invoice_items_table(engine, tablename, table_items_name, filepath, min_date):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        logger.info(f"Directory '{filepath}' created.")

    with engine.connect() as conn:
        max_date = datetime.date.today() + datetime.timedelta(days=31)
        dates_list = pd.date_range(min_date, max_date, freq='1ME')
        print(f"Date Range: {dates_list.min()} - {dates_list.max()}")

        for date in tqdm.tqdm(dates_list, desc="Invoice Items"):
            query = text(f"""
                SELECT a.*
                FROM MACDB.dbo.{table_items_name} a
                INNER JOIN (SELECT InvoiceID, ModifiedOn FROM MACDB.dbo.{tablename}) b
                ON a.InvoiceID = b.InvoiceID
                WHERE YEAR(b.ModifiedOn) = {date.year} AND MONTH(b.ModifiedOn) = {date.month}
            """)
            df = pd.read_sql(query, conn)
            if not df.empty:
                df.to_parquet(f"{filepath}/{date.year}{date.month:02}.parquet", index=False)

    return pd.read_parquet(filepath)

# --- INGESTION LOGIC ---
def ingest_promos():
    engine = _get_engine()
    return _get_all_data_from_table(engine, "TMP_PROMOS")

def ingest_customers():
    engine = _get_engine()
    customer_PE_df = _get_all_data_from_table(engine, "v_Customer")
    customer_PAC_df = _get_all_data_from_table(engine, "v_PAC_Customer")

    cols = customer_PE_df.columns
    customer_PE_df["StationBrand"] = "PE"
    customer_PAC_df["StationBrand"] = "PAC"

    customer_df = pd.concat([customer_PE_df, customer_PAC_df], axis=0).drop_duplicates(subset=cols)
    customer_df["Mobile"] = customer_df["Mobile"].str.zfill(10).str.slice_replace(stop=1, repl='966')
    
    out = customer_df.drop_duplicates(subset=cols)
    return out.astype(str)

def ingest_vehicles():
    engine = _get_engine()
    v_pe = _get_all_data_from_table(engine, "v_Vehicle")
    v_pac = _get_all_data_from_table(engine, "v_PAC_Vehicle")

    cols = v_pe.columns
    v_pe["StationBrand"], v_pac["StationBrand"] = "PE", "PAC"
    vehicles_df = pd.concat([v_pe, v_pac], axis=0).drop_duplicates(subset=cols)

    logger.info("Standardizing Vehicle Data...")
    vehicles_df["Make"] = vehicles_df["Make"].str.lower().str.replace("-", " ").str.replace(".", "", regex=False)
    vehicles_df["is_truck"] = vehicles_df["Make"].str.contains("truck").fillna(False).astype(int)

    # Dictionary for efficient bulk replacement
    make_map = {
        "cherry": "chery", "chevorlet": "chevrolet", "chevrolete": "chevrolet", "cheverolet": "chevrolet",
        "dihatsu": "daihatsu", "daihatzu": "daihatsu", "emegrand": "emgrand", "great wall": "gwm",
        "gelly": "geely", "hino 300": "hino", "300 hino": "hino", "hyudai": "hyundai", "hyundia": "hyundai",
        "hundai": "hyundai", "izusu": "isuzu", "izuzu": "isuzu", "range rover": "land rover",
        "masda": "mazda", "mazda6": "mazda", "mazda 6": "mazda", "mercedez": "mercedes",
        "mitshubishi": "mitsubishi", "mitsubushi": "mitsubishi", "pajero": "mitsubishi",
        "nisan": "nissan", "peugeut": "peugeot", "renult": "renault", "renualt": "renault",
        "zusuki": "suzuki", "camry": "toyota", "toyata": "toyota", "toyoya": "toyota"
    }
    
    for key, val in make_map.items():
        vehicles_df["Make"] = vehicles_df["Make"].str.replace(key, val, regex=False)

    # Simplified Model Adjustments (Example of your logic compressed)
    vehicles_df["Model"] = vehicles_df["Model"].str.lower().str.strip()
    
    # Pricing categorization
    price_levels = {
        "very_high": ["porsche", "lamborghini", "maserati", "bmw", "volvo", "jaguar", "mercedez", "chrysler", "dodge", "audi"],
        "high": ["toyota", "byd", "honda", "lexus", "jeep", "gmc", "lincoln"],
        "medium": ["volkswagen", "chevrolet", "fiat", "ford", "nissan", "mitsubishi", "mercury"],
        "low": ["pegout", "renault", "suzuki"],
        "very_low": ["jac", "chery"]
    }
    # Reverse the map for .map() function
    rev_price_map = {v: k for k, values in price_levels.items() for v in values}
    
    vehicles_df["vehicle_brand_level"] = vehicles_df["Make"].map(rev_price_map).fillna("other")
    vehicles_df["PlateNumber"] = vehicles_df["PlateNumber"].str.zfill(7)

    return vehicles_df.drop_duplicates(subset=cols).astype(str)

def ingest_branches():
    engine = _get_engine()
    b_pe = _get_all_data_from_table(engine, "v_Branch")
    b_pac = _get_all_data_from_table(engine, "v_PAC_Branch")

    cols = b_pe.columns
    b_pe["StationBrand"], b_pac["StationBrand"] = "PE", "PAC"
    df = pd.concat([b_pe, b_pac], axis=0).drop_duplicates(subset=cols)

    # Coordinate fix
    mask = df["Latitude"] > df["Longitude"]
    df.loc[mask, ["Latitude", "Longitude"]] = df.loc[mask, ["Longitude", "Latitude"]].values
    
    return df.drop_duplicates(subset=cols).astype(str)

def ingest_invoices():
    engine = _get_engine()
    logger.info("Downloading Invoices...")
    pe = get_all_data_from_invoice_table(engine, "v_Invoice")
    pac = get_all_data_from_invoice_table(engine, "v_PAC_Invoice")

    cols = pe.columns
    pe["StationBrand"], pac["StationBrand"] = "PE", "PAC"
    df = pd.concat([pe, pac], axis=0).drop_duplicates(subset=cols)
    df["InvoiceID"] = df["InvoiceID"].astype(str)
    
    return df.astype(str)

def ingest_invoices_items_PE(filepath, min_date):
    engine = _get_engine()
    df = get_all_data_from_invoice_items_table(engine, "v_Invoice", "v_InvoiceItems", filepath, min_date)
    return df.drop_duplicates().astype(str)

def ingest_invoices_items_PAC(filepath, min_date):
    engine = _get_engine()
    df = get_all_data_from_invoice_items_table(engine, "v_PAC_Invoice", "v_PAC_InvoiceItems", filepath, min_date)
    return df.drop_duplicates().astype(str)

def ingest_invoices_items(pe_df, pac_df):
    cols = pe_df.columns.tolist()
    pe_df["StationBrand"], pac_df["StationBrand"] = "PE", "PAC"
    
    df = pd.concat([pe_df, pac_df], axis=0).drop_duplicates(subset=cols)
    
    # Numeric conversions for math
    float_cols = ["ServiceTotalAmount", "ItemTotalAmount", "ServiceBeforeTaxAmount", "ItemBeforeTaxAmount", 
                  "ServiceBeforeDiscountAmount", "ItemBeforeDiscountAmount", "ServiceTotalDiscountAmount", 
                  "ItemTotalDiscountAmount", "ServiceItemCostAmount"]
    
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Logic calculations
    df["InvoiceTotalAmount"] = df["ServiceTotalAmount"] + df["ItemTotalAmount"]
    df["InvoiceGrossMargin"] = (df["ServiceBeforeTaxAmount"] + df["ItemBeforeTaxAmount"]) - df["ServiceItemCostAmount"]
    df["sku"] = df["ServiceName"].astype(str) + " | " + df["ServiceItemCode"].astype(str)
    
    return df.astype(str)

def ingest_transactions(invoice_df, items_df):
    logger.info("Merging and Processing Transactions...")
    
    # Ensure IDs match types
    invoice_df["InvoiceID"] = invoice_df["InvoiceID"].astype(str)
    items_df["InvoiceID"] = items_df["InvoiceID"].astype(str)

    # Filter columns to reduce memory usage on merge
    inv_cols = ["InvoiceID", "CustomerID", "CustomerVehicleID", "BranchID", "InvoiceDate", "WorkOrderMileage", "PreviousMileage"]
    item_cols = ["InvoiceID", "sku", "ItemBaseQuantity", "InvoiceTotalAmount", "InvoiceGrossMargin"]

    df = pd.merge(invoice_df[inv_cols], items_df[item_cols], on="InvoiceID", how="inner")
    
    # Final aggregation logic goes here...
    return df.astype(str)

# --- EXECUTION WRAPPERS ---
def ingestion_general():
    logger.info("Ingesting Master Data...")
    ingest_branches().to_parquet("data/01_raw/raw_branches.parquet", index=False)
    ingest_promos().to_parquet("data/01_raw/raw_promos.parquet", index=False)
    ingest_customers().to_parquet("data/01_raw/raw_customers.parquet", index=False)
    ingest_vehicles().to_parquet("data/01_raw/raw_vehicles.parquet", index=False)

def ingestion_invoices():
    ingest_invoices().to_parquet("data/01_raw/raw_invoices.parquet", index=False)

def main():
    # Setting the date to take fresh data from Feb 2026 onwards
    min_date = "2026-02-01"
    
    # UNCOMMENT these to actually pull fresh data from the SQL Server
    ingestion_general()
    ingestion_invoices()
    
    # Process Item Level
    path_pe = "data/01_raw/raw_invoices_items_PE_files"
    path_pac = "data/01_raw/raw_invoices_items_PAC_files"
    
    pe_items = ingest_invoices_items_PE(path_pe, min_date)
    pac_items = ingest_invoices_items_PAC(path_pac, min_date)
    
    final_items = ingest_invoices_items(pe_items, pac_items)
    final_items.to_parquet("data/01_raw/raw_invoices_items.parquet", index=False)

    logger.info("Pipeline Complete.")

if __name__ == "__main__":
    main()