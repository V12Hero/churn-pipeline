# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.
"""Feature utils."""
import logging
from typing import List

import dateutil
import numpy as np
import pandas as pd

from transactions.nodes import add_days_since_last_transaction

logger = logging.getLogger(__name__)


def filter_last_n_days_or_months(
    df: pd.DataFrame,
    num_days: int,
    convert_to_months: bool = True,
    from_reference_date: bool = False,
) -> pd.DataFrame:
    """Filter last N days or months from the evaluation date.

    Note that on the timeline the right side included, left side not included.
    Ex: for '2023-12-01' and 3 months we will have '2023-12-01', '2023-11-01'
    and '2023-10-01'.

    If convert_to_months is set to True the number os days will be round to months. Ex:
    "num_days": [90] becomes 3 months and the filter is done in whole months.

    If from_reference_date is set to True will filter the last num_days based on the
    reference date. If not, it will take the last 90 days of the table.

    Args:
        df(pd.Dataframe): Dataframe with transactions data.
        num_days(int): Number of days to filter.
        convert_to_months(bool): If true convert number of days to months and the filter
                                 is done in whole months.
        from_reference_date(bool): If true the filter is done from the reference date,
                                   if not is done with the max date.

    Returns:
        Dataframe filtered.
    """
    # TODO: Create other cases
    if convert_to_months:
        assert num_days % 30 == 0, (
            f"If convert_to_months=True, num_days must be "
            f"divisible by 30, but got num_days={num_days}."
        )
        n_months = num_days // 30

    df = _trunc_month(
        df, date_col_name="order_delivered_at_date", month_col_name="order_date_month"
    )

    if from_reference_date:
        df = _trunc_month(
            df, date_col_name="evaluation_date", month_col_name="evaluation_date_month"
        )
        df['nb_months'] = (
            df["evaluation_date_month"] - df["order_date_month"]
        ) / np.timedelta64(1, 'M')
        df_filterd = df[df['nb_months'] < n_months]
    else:
        max_month = df[["order_date_month"]].max()[0]
        min_month = max_month - dateutil.relativedelta.relativedelta(months=n_months)
        df_filterd = df[
            (df["order_date_month"] <= max_month) & (df["order_date_month"] > min_month)
        ]

    return df_filterd


def get_last_order(df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
    """Get the last order data.

    function creates "last_order_date", "last_order_sales_amount", "last_order_quantity"
    and "days_since_last_order" columns by getting the max value of
    "order_delivered_at_date" as the last order and pivoting.

    Args:
        df (pd.Dataframe): Dataframe with transactions data.
        groupby_cols(List[str]): Columns used as spine to group by the data.

    Returns:
        Dataframe with `groupby_cols` plus "last_order_date", "last_order_sales_amount",
        "last_order_quantity" and "days_since_last_order".
    """
    df_last_order_spine = (
        df.groupby(groupby_cols).agg({"order_delivered_at_date": "max"}).reset_index()
    )

    df_last_order = pd.merge(
        df,
        df_last_order_spine,
        on=groupby_cols + ["order_delivered_at_date"],
        how="inner",
    )

    df_last_order = df_last_order.rename(
        columns={
            "order_delivered_at_date": "last_order_date",
            "sales_amount": "last_order_sales_amount",
            "quantity": "last_order_quantity",
        }
    )

    df_last_order = df_last_order[
        groupby_cols
        + ["last_order_date", "last_order_sales_amount", "last_order_quantity"]
    ]

    df_last_order['days_since_last_order'] = (
        df_last_order['evaluation_date'] - df_last_order['last_order_date']
    ) / np.timedelta64(1, 'D')
    return df_last_order


def agg_total_and_avg(df: pd.DataFrame, groupby_cols: List[str]):
    """Agg total and average values.

    The mean and sum columns are named like: "sales_amount_sum", "sales_amount_mean"

    Args:
        df (pd.Dataframe): Dataframe with columns 'sales_amount', 'quantity' and
                           'transaction_id'.
        groupby_cols(List[str]): Columns to group by. Ex: ['customer_id', 'sku']

    Returns:
        Dataframe grouped by `groupby_cols` and sum and mean values calculated.
    """
    df_agg = (
        df.groupby(groupby_cols)
        .agg(
            {
                "sales_amount": ["sum", "mean"],
                "quantity": ["sum", "mean"],
                "transaction_id": "count",
            }
        )
        .reset_index()
    )

    df_agg.columns = df_agg.columns.to_flat_index()
    df_agg.columns = ['_'.join(col) for col in df_agg.columns.values]
    df_agg.columns = [col[:-1] if col.endswith("_") else col for col in df_agg.columns]

    return df_agg


def _trunc_month(
    df: pd.DataFrame, date_col_name: str, month_col_name: str
) -> pd.DataFrame:
    """Trunc date col to month.

    If both date_col_name and month_col_name are equal the column is truncated in place.
    If not a new column is created.

    Args:
        df (pd.DataFrame): Pandas Dataframe with a column named with the value of
                           `date_col_name`.
        date_col_name (str): Name of the date col that will be truncated to month.
        month_col_name (str): Name of the new column with the date truncated to month.

    Returns:
        Pandas dataframe with a new column for the truncated date.

    """
    df[month_col_name] = (
        pd.to_datetime(df[date_col_name]).dt.to_period('M').dt.to_timestamp()
    )
    return df


def average_days_between_orders(
    df: pd.DataFrame,
    transaction_date_col: str,
    customer_id_col: str,
):
    """Calculate days between orders for customer ids.

    Args:
        df: Dataframe with `customer_id_col` and `transaction_date_col`
        transaction_date_col: Date column used as reference for transactions date.
                                Ex: 'order_delivered_at_date'.
        customer_id_col: Column used as reference for customer id. Ex: "customer_id"

    Returns:
        Dataframe with a new column `days_between_col`

    """
    params = {
        "cols": {
            "transaction_date": transaction_date_col,
            "customer_id": customer_id_col,
            "days_between": "days_between_col",
        }
    }
    df = df[[customer_id_col, transaction_date_col]]

    df_days_between = add_days_since_last_transaction(df, params)

    df_out = (
        df_days_between.groupby(customer_id_col)
        .agg(average_days_between_transactions=pd.NamedAgg("days_between_col", "mean"))
        .reset_index()
    )
    return df_out
