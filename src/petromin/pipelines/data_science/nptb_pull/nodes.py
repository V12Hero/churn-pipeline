"""
This is a boilerplate pipeline 'nptb_pull'
generated using Kedro 0.18.12
"""

import pyspark
from typing import List, Union, Dict
import pyspark.sql.functions as f
from dateutil.relativedelta import relativedelta
from pyspark.sql.window import Window
import pandas as pd

def nptb_clean_transactions_data(
        prm_spine: pyspark.sql.DataFrame,
        int_transactions:pyspark.sql.DataFrame,
        categories_to_drop: List,
        products_to_drop: List
):
    # sectors_to_drop = list(set(sectors_to_drop).union(set(sectors_to_drop_additional)))
    # relevant_ids = master.filter(
    #     ~master.sector_id_original.isin(sectors_to_drop)
    # ).select(
    #     f.col('_id').alias('client_id')
    # ).dropDuplicates()
    # transactions = transactions.join(relevant_ids, on='client_id', how='inner')
    # transactions = transactions.filter(
    #     ~(
    #     ((f.col('transaction_typ')=='FA') & (f.col('total_product_val')<=0))|
    #     (f.col('total_catalog_product_val')<f.col('total_product_val'))
    #     )
    # )

    # products = products.withColumn(
    #     "category_name",
    #     f.regexp_replace(f.lower(f.col("category_name")), ' ', '_')
    # ).drop('product_name')

    # products = products.filter(~f.col('product_code').isin(products_to_drop))

    # transactions = transactions.join(
    #     products,
    #     on=["product_code"],
    #     how="left"
    # )
    # categories_to_drop = list(set(categories_to_drop)-set(categories_to_keep))
    # transactions = transactions.withColumn(
    #     'category_name',
    #     f.when(
    #         (f.col('category_name')=='premios') & 
    #         ((f.col('total_product_val')>1) | (f.col('total_catalog_product_val')>1)),
    #         f.lit('others')
    #     ).otherwise(f.col('category_name'))
    # )
    # transactions = transactions.filter(~f.col('category_name').isin(categories_to_drop))
    # transactions = transactions.withColumn('year_month', f.col('transaction_dt').substr(0,7))

    int_transactions = int_transactions.withColumn(
        "_key", 
        f.concat_ws("__", *["customer_id", "customer_vehicle_id"])
    ).withColumn(
        "_observ_end_dt",
        f.last_day(f.col("transaction_dt"))
    )

    base_transactions = prm_spine.join(
        int_transactions,
        on=["_key", "_observ_end_dt"],
        how="inner"
    ).filter(
        ~f.col('product_category').isin(categories_to_drop)
    ).filter(
        ~f.col('product_code').isin(products_to_drop)
    )

    return base_transactions

def prepare_transactions(
        base_transactions:pyspark.sql.DataFrame,
        n_months_nptb_pull: int,
        initial_date,

):

    # initial_date_n_months_before = (pd.to_datetime(initial_date) + pd.offsets.MonthEnd(0)) - relativedelta(months=n_months_nptb_pull)
    initial_date_n_months_before = pd.to_datetime(initial_date) - relativedelta(months=n_months_nptb_pull)

    transactions = base_transactions.filter(
        f.col("_observ_end_dt") >= initial_date_n_months_before
    ).select(
        "_observ_end_dt",
        "_id",
        "customer_id",
        "customer_vehicle_id",
        "product_category",
        "product_code",
        "quantity",
        "sales_amount_net",
        "total_profit",
    ).groupBy(
        "_observ_end_dt", "_id", "customer_id", "customer_vehicle_id", "product_category", "product_code",
    ).agg(
        f.sum('quantity').alias('product_qty'),
        f.sum('sales_amount_net').alias('total_product_val'),
        f.sum('total_profit').alias('total_profit_product_val')
    ).orderBy("_id", "product_category", "product_code", "_observ_end_dt")

    return transactions

def nptb_pull_create_features(
        prm_spine:pyspark.sql.DataFrame,
        transactions:pyspark.sql.DataFrame,
        n_months_nptb_pull: int,
        n_recommendations_pull: int,
        min_number_of_trx_for_recommendations: int,
):

    window_client_product = Window.partitionBy(['_id', 'product_code']).orderBy(f.col("_observ_end_dt"))
    window_client = Window.partitionBy(['_id']).orderBy(f.col("_observ_end_dt"))

    date_range = prm_spine.select('_observ_end_dt').distinct().orderBy('_observ_end_dt')
    min_date_per_client = transactions.groupby('_id', 'product_code').agg(f.min('_observ_end_dt').alias('first_trx_dt'))

    max_date = date_range.agg({"_observ_end_dt": "max"}).collect()[0]["max(_observ_end_dt)"]

    new_month = date_range.filter(
        date_range._observ_end_dt == max_date
    ).withColumn(
        '_observ_end_dt',
        f.add_months(f.col('_observ_end_dt'), 1)
    )

    new_date = new_month.agg({"_observ_end_dt": "max"}).collect()[0]["max(_observ_end_dt)"]

    date_range = date_range.union(new_month).orderBy('_observ_end_dt')

    #

    client_sku = transactions.select(['_id', 'product_code',]).distinct()
    client_sku_first_dt = client_sku.join(
        min_date_per_client,
        on=['_id', 'product_code'],
        how='inner'
    )

    sku_date_crossjoin = client_sku_first_dt.crossJoin(
        f.broadcast(date_range)
    ).filter(
        f.col('_observ_end_dt') >= f.col('first_trx_dt')
    ).drop('first_trx_dt')

    transactions_after_fst_trx = transactions.join(
        sku_date_crossjoin,
        on=['_id', 'product_code', '_observ_end_dt'],
        how='right'
    )

    #

    transactions_per_client = transactions_after_fst_trx.groupBy(
        ['_id', '_observ_end_dt']
    ).agg(
        f.sum('total_product_val').alias('total_client_net_sales'),
        f.sum('total_profit_product_val').alias('total_client_profit_sales')
    ).withColumn(
        'placed_order', 
        f.when(f.col('total_client_net_sales') > 0, 1).otherwise(0)
    ).withColumn(
        'total_client_orders_last_months',
        f.sum('placed_order').over(window_client.rowsBetween(-n_months_nptb_pull, -1))
    ).withColumn(
        'total_client_net_sales_last_months',
        f.sum('total_client_net_sales').over(window_client.rowsBetween(-n_months_nptb_pull, -1))
    ).withColumn(
        'total_client_profit_sales_last_months',
        f.sum('total_client_profit_sales').over(window_client.rowsBetween(-n_months_nptb_pull, -1))
    ).drop('placed_order')

    ftr_transactions = transactions_after_fst_trx.join(
        transactions_per_client,
        on=['_id', '_observ_end_dt'],
        how='left'
    ).withColumn(
        'last_transaction_month',
        f.when(
            f.col('total_product_val').isNotNull(),
            f.col('_observ_end_dt')
        ).otherwise(None)
    ).withColumn(
        'last_transaction_month',
        f.last('last_transaction_month', ignorenulls=True).over(window_client_product.rowsBetween(Window.unboundedPreceding, -1))
    ).withColumn(
        'total_product_val_last_order',
        f.last('total_product_val', ignorenulls=True).over(window_client_product.rowsBetween(Window.unboundedPreceding, -1))
    ).withColumn(
        'total_profit_product_val_last_order',
        f.last('total_profit_product_val', ignorenulls=True).over(window_client_product.rowsBetween(Window.unboundedPreceding, -1))
    ).withColumn(
        'units_last_order',
        f.last('product_qty', ignorenulls=True).over(window_client_product.rowsBetween(Window.unboundedPreceding, -1))
    ).withColumn(
        'months_from_last_order', #months passed since last order
        f.months_between(f.col('_observ_end_dt'),f.col('last_transaction_month'))
    ).withColumn(
        'months_between_order', #only exists for months where order was placed
        f.when(
            f.col('total_product_val').isNotNull(),
            f.months_between(f.col('_observ_end_dt'), f.col('last_transaction_month'))
        ).otherwise(None)
    ).withColumn(
        'avg_units_per_order_last_months',
        f.mean('product_qty').over(window_client_product.rowsBetween(-n_months_nptb_pull, -1))
    ).withColumn(
        'avg_months_between_orders', 
        f.mean('months_between_order').over(window_client_product.rowsBetween(-n_months_nptb_pull, -1))
    ).withColumn(
        'n_orders_sku_last_months',
        f.count('total_product_val').over(window_client_product.rowsBetween(-n_months_nptb_pull, -1))
    ).withColumn(
        'net_sales_sku_last_months', 
        f.sum('total_product_val').over(window_client_product.rowsBetween(-n_months_nptb_pull, -1))
    ).withColumn(
        'profit_sales_sku_last_months',
        f.sum('total_profit_product_val').over(window_client_product.rowsBetween(-n_months_nptb_pull, -1))
    ).withColumn(
        'sales_contribution_net',
        f.col('net_sales_sku_last_months') / f.col('total_client_net_sales_last_months')
    ).withColumn(
        'sales_contribution_catalog',
        f.col('profit_sales_sku_last_months') / f.col('total_client_profit_sales_last_months')
    ).withColumn(
        'repetition_rate',
        f.col('n_orders_sku_last_months') / f.col('total_client_orders_last_months')
    ).withColumn(
        'days_expected_breakage',
        f.col('units_last_order') / (f.col('avg_units_per_order_last_months') / f.col('avg_months_between_orders'))
    ).withColumn(
        'breakage_rate',
        f.col('months_from_last_order') / f.col('days_expected_breakage')
    ).withColumn(
        'adjusted_breakage_rate',
        f.when(
            f.col('breakage_rate') >= 1,
            f.col('breakage_rate')
        ).otherwise(0.1)
    ).withColumn(
        'prioritization_index',
        (f.col('months_from_last_order') / f.col('avg_months_between_orders')) * f.col('sales_contribution_catalog') * f.col('repetition_rate'))

    # breakpoint()

    window_prioritization = Window.partitionBy(['_id', '_observ_end_dt']).orderBy(f.col("prioritization_index").desc())

    ftr_transactions_filtered = ftr_transactions.filter(
        f.col("prioritization_index").isNotNull()
    ).withColumn(
        'ranking', 
        f.row_number().over(window_prioritization)
    ).filter(
        f.col("_observ_end_dt") >= new_date
    ).filter(
        f.col("total_client_orders_last_months") >= min_number_of_trx_for_recommendations
    ).filter(
        f.col("ranking") <= n_recommendations_pull
    ).withColumnRenamed(
        'product_code',
        'product_code_pull'
    )

    out = ftr_transactions_filtered.orderBy(
        ['_id', 'product_code', '_observ_end_dt']
    ).select(
        '_id',
        'product_code_pull',
        'ranking'
    )

    return out
