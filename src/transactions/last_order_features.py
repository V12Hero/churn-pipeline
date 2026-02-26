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
"""Last order features."""
import pandas as pd

from transactions.utils.feature_utils import (
    agg_total_and_avg,
    average_days_between_orders,
    filter_last_n_days_or_months,
    get_last_order,
)


def last_order_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Create features for the pull and push models.

    Parameters:
        df: Pandas dataframe with at least the following columns:
            - *params["groupby_cols"]: Columns from the 'params' variable.
            - evaluation_date
            - order_date_month
            - order_delivered_at_date
            - sales_amount
            - quantity


    Create the following features:
    Pull:
    - Days from last order in the last xx days
    - Sales from last order in the last xx days
    - Volume from last order in the last xx days
    - Average orders in the last xx days
    - Average sales in the last xx days
    - Average volume in the last xx days

    Push:
    - Total sales in the last xx days
    - Total orders in the last xx days
    - Total volume in the last xx days
    """
    # TODO: Validate params dict or open parameters on function
    # TODO: Check datatypes
    # TODO: This cast is a temporary solution.
    # Data type validation shouldn't be done on feature creation, but before.
    df['evaluation_date'] = df['evaluation_date'].astype('datetime64[ns]')
    df['order_delivered_at_date'] = df['order_delivered_at_date'].astype(
        'datetime64[ns]'
    )

    df_filtered = filter_last_n_days_or_months(
        df=df,
        num_days=params["num_days"][0],
        convert_to_months=True,
        from_reference_date=False,
    )
    df_last_order = get_last_order(df=df, groupby_cols=params["groupby_cols"])
    df_agg = agg_total_and_avg(df=df_filtered, groupby_cols=params["groupby_cols"])
    df_days_between_orders = average_days_between_orders(
        df=df,
        transaction_date_col="order_delivered_at_date",
        customer_id_col="customer_id",
    )

    out_df = pd.merge(df_last_order, df_agg, on=params["groupby_cols"], how="left")
    out_df = pd.merge(out_df, df_days_between_orders, on="customer_id", how="left")

    out_df = out_df.rename(
        columns={
            "sales_amount_sum": f"sales_amount_sum_last_{params['num_days'][0]}_days",
            "sales_amount_mean": f"sales_amount_mean_last_{params['num_days'][0]}_days",
            "quantity_sum": f"quantity_sum_last_{params['num_days'][0]}_days",
            "quantity_mean": f"quantity_mean_last_{params['num_days'][0]}_days",
            "transaction_id_count": f"num_orders_last_{params['num_days'][0]}_days",
            "average_days_between_transactions": f"average_days_between_transactions_last_{params['num_days'][0]}_days",
        }
    )
    # TODO: Check NaN on out_df

    return out_df
