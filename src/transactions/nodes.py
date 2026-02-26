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

import logging
from operator import itemgetter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from segmentation_core.helpers.data_processing.general import clean_string, join_dfs
from segmentation_core.helpers.objects import _load_obj
from segmentation_core.helpers.qa import check_duplicates

logger = logging.getLogger(__name__)


@check_duplicates(["transaction_id", "sku"])
def prepare_transactional_data(
    df: pd.DataFrame,
    params: Dict,
) -> pd.DataFrame:
    """Add specific information to the transactional data.

    - Discount information
    - Evaluation dates to iterate the feature creation
    - Number of days since last transaction information

    Args:
      df (pd.DataFrame): the dataframe to be transformed
      params (Dict): Dictionary with parameters
    Returns:
      A dataframe with the new information
    """
    if "sample" in params:
        df = df.sample(params["sample"], random_state=42)

    logger.info("Adding discount information column")
    df = _add_discount_information(df, params)
    logger.info("Adding month and week")
    df = _add_week_month(df, params)
    #
    print(params)
    logger.info("Adding evaluation date column")
    df = _add_evaluation_date_col(df, params)
    logger.info("Adding days since last transaction column")
    df = add_days_since_last_transaction(df, params)
    return df


def _add_discount_information(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """It adds discount information to the dataframe.

    The information is in the columns:
    - has_discount_col: indicates whether the item had a discount or not
    - product_id_col: product id that had discount discount
    - transaction_id_col: transaction_id that had discount
    - quantity_col_discount: quantity purchased that had discount

    Args:
      df (pd.DataFrame): The dataframe that we're working with
      params (dict): Dictionary with parameters

    Returns:
      A dataframe with the following columns:
        - has_discount_col
        - product_id_col_discount
        - transaction_id_col_discount
        - quantity_col_discount
    """
    has_discount_col, product_id_col, transaction_id_col, quantity_col = itemgetter(
        "has_discount", "product_id", "transaction_id", "quantity"
    )(params["cols"])

    df[has_discount_col] = df["sum_discounted_value"].apply(
        lambda x: True if x > 0 else False
    )

    # Adding specific columns for discount vs non discount items
    df.loc[df[has_discount_col], f"{product_id_col}_discount"] = df.loc[
        df[has_discount_col], product_id_col
    ]
    df.loc[df[has_discount_col], f"{transaction_id_col}_discount"] = df.loc[
        df[has_discount_col], transaction_id_col
    ]
    df[f"{quantity_col}_discount"] = df[quantity_col] * df[has_discount_col]

    return df


def _add_week_month(df: pd.DataFrame, params: Dict):
    """
    > Add month and week columns to the dataframe
    Args:
      df (pd.DataFrame): The dataframe that we're going to be adding the week and month columns to.
      params (Dict): Dictionary with parameters
    Returns:
      A dataframe with the columns "month" and "week" added.
    """
    transaction_date_col = params["cols"]["transaction_date"]
    df["month"] = (
        pd.DatetimeIndex(df[transaction_date_col]).to_period("M").to_timestamp()
    )
    df["week"] = (
        pd.DatetimeIndex(df[transaction_date_col]).to_period("W").to_timestamp()
    )
    df["week"] = df["week"].apply(lambda x: str(x).split("/")[0])

    return df


def _add_evaluation_date_col(df: pd.DataFrame, params: dict):
    """Adds column with evaluation days to be the reference to calculate temporal features
    Args:
        df (DataFrame): input dataset
        params (dict): Dictionary with parameters
    Returns:
        DataFrame
    """
    start = params["evaluation_date_start"]
    end = params["evaluation_date_end"]
    freq = params["freq"]
    transaction_date_col = params["cols"]["transaction_date"]
    new_df = df.copy()

    days_since_col, transaction_date_col = itemgetter(
        "days_since",
        "transaction_date",
    )(params["cols"])

    # [define evaluation dates]
    if start is None:
        start = new_df[transaction_date_col].max()
    if end is not None:
        # Cross join to duplicate transactions for each evaluation date
        evaluation_dates = pd.date_range(start=start, end=end, freq=freq)
    else:
        evaluation_dates = [start]
    if (
        "manual_evaluation_dates" in params
        and len(params["manual_evaluation_dates"]) > 0
    ):
        evaluation_dates = params["manual_evaluation_dates"]

    # [outer join to have all transactions at different evaluation dates]
    new_df["tmp"] = 1.0
    logger.info(f"Evaluation dates: {evaluation_dates}")
    new_df = new_df.merge(
        pd.DataFrame(
            {
                "evaluation_date": evaluation_dates,
                "tmp": np.ones(len(pd.date_range(start=start, end=end, freq=freq))),
            }
        ),
        on="tmp",
        how="outer",
    ).drop("tmp", axis=1)

    # Filter past transactions for each evaluation date
    new_df[transaction_date_col] = pd.to_datetime(new_df[transaction_date_col])
    new_df["evaluation_date"] = pd.to_datetime(new_df["evaluation_date"])
    new_df[days_since_col] = (
        new_df["evaluation_date"] - new_df[transaction_date_col]
    ).dt.days

    new_df = new_df[new_df[days_since_col] >= 0]

    return new_df


def add_days_since_last_transaction(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Adds column with days since last transaction.

    Args:
        df (pd.DataFrame): transactions data.
        params (Dict): Dictionary with parameters
    Returns:
        pd.DataFrame: transactions data with days since last transaction.
    """
    days_between_col, transaction_date_col, customer_id = itemgetter(
        "days_between", "transaction_date", "customer_id"
    )(params["cols"])

    df[days_between_col] = (
        df.sort_values(by=[customer_id, transaction_date_col])
        .groupby(customer_id)[transaction_date_col]
        .diff()
        .dt.days
    )
    return df


def build_transactional_features_df(
    df: pd.DataFrame,
    groupby_cols: List[str],
    features_params: Dict,
    num_days_list: List[int],
    global_params: Dict,
):
    """Calculates several transaction features for.

    - specific time windows specified in the num_days_list
    - overall features for the entire transactional data history

    Args:
        df (DataFrame): input dataset
        features (List[str]): List of feature groups to create
        groupby_cols (List[str]): List of columns to groupby
        num_days_list (List[int], optional): List with number of days to calculate the features.
        params (Dict): Dictionary with parameters.
    Returns:
        pd.DataFrame
    """
    cols_selected = [
        "days_since",
        "days_between",
        "transaction_id",
        "product_id",
        "quantity",
        "sales_amount",
        "discount",
        "product_category",
        "gross_profit",
    ]
    cols = {col: global_params["cols"][col] for col in cols_selected}
    transaction_date_col = cols["days_since"]

    users_no = df["customer_id"].drop_duplicates().shape[0]

    final_dfs_list = []

    logger.info(f"Building transactional features for {groupby_cols}")
    for day in df["evaluation_date"].unique():
        # Calculating "window" features for every window in num_days_list
        window_dfs = []
        for num_days in num_days_list:
            logger.info(
                f"Calculating window transactional features for {num_days} days window starting on {day}"
            )
            assert num_days % 30 == 0
            # [filt dataframe based on months ]
            n_months = num_days // 30
            sel_months = (
                df[["month"]].drop_duplicates().sort_values(by="month").tail(n_months)
            )
            filt_df = df[(df["evaluation_date"] == day)]
            filt_df = sel_months.merge(filt_df, how="left", on="month")

            logger.info(n_months)
            logger.info(filt_df.groupby("month").size())

            min_date = str(filt_df[transaction_date_col].min())
            max_date = str(filt_df[transaction_date_col].max())

            logger.info("n_months")
            logger.info(f"{min_date} {max_date}")

            new_df = _calculate_transactional_features(
                df=filt_df,
                groupby_cols=groupby_cols,
                num_days=num_days,
                window=True,
                features=features_params,
                cols=cols,
            )
            logger.info(f"new_df shape: {new_df.shape}")
            window_dfs.append(new_df)

        # Joining dfs for different windows
        window_df = join_dfs(groupby_cols[0], *window_dfs)
        window_df.reset_index(inplace=True)
        # Calculating "general" features
        logger.info(f"Calculating general transactional features for {num_days} days")
        general_df = _calculate_transactional_features(
            df=df,
            groupby_cols=groupby_cols,
            num_days="",
            window=False,
            features=features_params,
            cols=cols,
        )

        # Joining general and window features
        tmp_df = join_dfs(groupby_cols[0], general_df, window_df)
        tmp_df["evaluation_date"] = day

    # Joining dfs for different evaluation dates
    final_dfs_list.append(tmp_df)
    return_df = pd.concat(final_dfs_list)

    # Treating nan columns for transactional features
    logger.info("Treating null columns")
    df = _nan_treatment_transactional(return_df, False)

    for col in df.columns:
        if col.startswith("index"):
            df.drop(col, axis=1, inplace=True)

    if "regex" in features_params:
        columns = [groupby_cols[0], "evaluation_date"]
        feat_cols = list(df.filter(regex=features_params["regex"]).columns)
        columns += feat_cols

        df = df[columns]

        prefix = features_params.get("prefix", None)

        if prefix:
            df.columns = [
                prefix + col if col in feat_cols else col for col in df.columns
            ]
        logger.info(f"Adding prefix: {prefix}")

    users_no_final = df["customer_id"].drop_duplicates().shape[0]

    assert users_no_final == users_no

    # Update column names in DataFrame
    df.columns = [clean_string(col) for col in df.columns]

    logger.info(f"Filters: {features_params}")
    logger.info(f"Resulting shape: {df.shape}")

    return df


def _calculate_transactional_features(
    df: pd.DataFrame,
    groupby_cols,
    num_days,
    cols: Dict,
    window,
    features={},
):
    """It calculates the features for different time period/granularities
    (weekly, per order, and overall) and then joins them together.

    Args:
      df (pd.DataFrame): the dataframe containing the transactional data
      groupby_cols: The columns to group by.
      num_days: The number of days to look back if calculating window features.
      days_since_col (Optional[str]): The column that contains the number of days since the last
    window. Defaults to True

    Returns:
      A dataframe with the transactional features.
    """
    features_arr = []
    transaction_id_col = itemgetter(
        "transaction_id",
    )(cols)

    granularity_map = {"week": "week", "month": "month", "order": transaction_id_col}
    for suffix in features.get("granularities", []):
        granularity_col = granularity_map[suffix]
        ftr_i = _calculate_features_different_granularities(
            df=df,
            groupby_cols=groupby_cols,
            granularity_col=granularity_col,
            suffix=suffix,
            cols=cols,
        )
        features_arr.append(ftr_i)

    if features.get("overall", False):
        logger.info("Calculating general features")
        transaction_general_features = _calculate_overall_features(
            df, groupby_cols, cols
        )
        features_arr.append(transaction_general_features)

    if features.get("frequency", False):
        logger.info("Calculating frequency features")
        frequency_features = _calculate_transaction_frequency_features(
            df, groupby_cols, features["frequency"], cols  # conf
        )
        features_arr.append(frequency_features)

    new_df = join_dfs(groupby_cols, *features_arr)

    final_df = _pivot_and_rename(new_df, groupby_cols)

    if window and num_days != "":
        logger.info("Renaming columns")
        rename_cols = {
            col: f"{col}_last_{num_days}_days"
            for col in final_df.columns
            if col not in groupby_cols
        }

        final_df.rename(columns=rename_cols, inplace=True)

    return final_df


def _calculate_features_different_granularities(
    df: pd.DataFrame, groupby_cols, granularity_col, suffix, cols: Dict
):
    """It calculates the following transactional features for different
    specific granularities (most common are week and per order)

    - mean_distinct_sku: Number of distinct SKUs purchased
    - mean_distinct_sku_discount: Number of distinct SKUs purchased with discount
    - mean_gross_profit: Average gross profit
    - mean_mix_category: Number/mix of distinct categories purchased
    - mean_sales_amount: Average sales amount
    - mean_sales_quantity: Average sales quantity
    - mean_discount: Average discount
    - mean_value_promo_share: Average "promo share" (discount/sales amount)
    - mean_distinct_sku_promo_share: Average SKU "promo share" (SKUs with discount/total SKUs)

    It also calculates the weighted average of gross margin features

    Args:
      df (pd.DataFrame): the dataframe you want to calculate the features on
      groupby_cols: the columns to group by
      granularity_col: the column that you want to group by. For example, if you want to group by month,
    then this column should be the month column.
      suffix: This is the suffix that will be added to the end of the column names.
      product_id_col (Optional[str]): Optional[str] = "sku",. Defaults to sku
      quantity_col (Optional[str]): The column name of the quantity column in the dataframe. Defaults to

    Returns:
      A dataframe with the new trasactional columns
    """
    (
        product_id_col,
        quantity_col,
        sales_amount_col,
        discount_col,
        product_category_col,
        gross_profit_col,
    ) = itemgetter(
        "product_id",
        "quantity",
        "sales_amount",
        "discount",
        "product_category",
        "gross_profit",
    )(
        cols
    )

    df_before_mean = (
        df.groupby(groupby_cols + [granularity_col])
        .agg(
            mean_distinct_sku=pd.NamedAgg(product_id_col, "nunique"),
            mean_distinct_sku_discount=pd.NamedAgg(
                f"{product_id_col}_discount", "nunique"
            ),
            mean_gross_profit=pd.NamedAgg(gross_profit_col, "sum"),
            mean_mix_category=pd.NamedAgg(product_category_col, "nunique"),
            mean_sales_amount=pd.NamedAgg(sales_amount_col, "sum"),
            mean_sales_quantity=pd.NamedAgg(quantity_col, "sum"),
            mean_discount=pd.NamedAgg(discount_col, "sum"),
        )
        .reset_index()
    )

    df_before_mean["mean_value_promo_share"] = df_before_mean["mean_discount"] / (
        df_before_mean["mean_sales_amount"] + df_before_mean["mean_discount"]
    )
    df_before_mean["percentage_distinct_sku_with_discount"] = (
        df_before_mean["mean_distinct_sku_discount"]
        / df_before_mean["mean_distinct_sku"]
    )

    df_mean = df_before_mean.groupby(groupby_cols).mean().reset_index()

    final_df = join_dfs(groupby_cols, df_mean)

    rename_cols = {
        col: f"{col}_per_{suffix}"
        for col in final_df.columns
        if col not in groupby_cols
    }
    final_df.rename(columns=rename_cols, inplace=True)

    return final_df


def _calculate_overall_features(
    df: pd.DataFrame,
    groupby_cols: List[str],
    cols: Dict,
):
    """It takes a dataframe, a list of columns to group by, and two optional
    columns to use as the product id and product category, and returns a
    dataframe with the total number of distinct products, the total number of
    distinct products with a discount, the total number of distinct product
    categories, and the percentage of distinct products with a discount.

    Args:
      df (pd.DataFrame): pd.DataFrame - the dataframe you want to calculate the features on
      groupby_cols (List[str]): List of columns to groupby.
      product_id_col (Optional[str]): The column name of the product id. Defaults to sku
      product_category_col (Optional[str]): The column name of the product category. Defaults to sku_category
    """
    product_id_col, product_category_col = itemgetter("product_id", "product_category")(
        cols
    )
    general_df = (
        df.groupby(groupby_cols)
        .agg(
            total_distinct_sku=pd.NamedAgg(product_id_col, "nunique"),
            total_distinct_sku_discount=pd.NamedAgg(
                f"{product_id_col}_discount", "nunique"
            ),
            total_mix_category=pd.NamedAgg(product_category_col, "nunique"),
        )
        .reset_index()
    )

    general_df["percentage_distinct_sku_with_discount"] = (
        general_df["total_distinct_sku_discount"] / general_df["total_distinct_sku"]
    )

    return general_df


def _calculate_transaction_frequency_features(
    df: pd.DataFrame,
    groupby_cols: List[str],
    conf: List[str],
    cols: Dict,
    active_user_window: Optional[int] = 30,
):
    """It takes a dataframe, a list of columns to group by, and a few other
    optional parameters, and returns a dataframe with transactional features
    related to the purchase frequency.

    Args:
      df (pd.DataFrame): the dataframe you want to calculate the features on
      groupby_cols (List[str]): The columns to group by.
      transaction_id_col (Optional[str]): The column that contains the transaction id. Defaults to

    Returns:
      A dataframe with the following columns:
        - days_since_last_transaction
        - days_since_first_transaction
        - average_days_between_transactions
        - number_of_transactions
        - number_of_transactions_with_discount
        - percentage_of_transactions_with_discount
    """
    features_arr = []
    transaction_id_col, days_since_col, days_between_col = itemgetter(
        "transaction_id", "days_since", "days_between"
    )(cols)
    if "monthly" in conf:
        df_per_month = _calculate_frequency_features(
            df, groupby_cols, cols, "month", "M"
        )
        features_arr.append(df_per_month)
    if "weekly" in conf:
        df_per_week = _calculate_frequency_features(df, groupby_cols, cols, "week", "W")
        features_arr.append(df_per_week)
    if "daily" in conf:
        df_days_frequency = (
            df.groupby(groupby_cols)
            .agg(
                days_since_last_transaction=pd.NamedAgg(days_since_col, "min"),
                days_since_first_transaction=pd.NamedAgg(days_since_col, "max"),
                average_days_between_transactions=pd.NamedAgg(days_between_col, "mean"),
                number_of_transactions=pd.NamedAgg(transaction_id_col, "nunique"),
                number_of_transactions_with_discount=pd.NamedAgg(
                    f"{transaction_id_col}_discount", "nunique"
                ),
            )
            .reset_index()
        )

        df_days_frequency["percentage_of_transactions_with_discount"] = (
            df_days_frequency["number_of_transactions_with_discount"]
            / df_days_frequency["number_of_transactions"]
        )

        df_days_frequency.loc[
            df_days_frequency["days_since_last_transaction"] <= active_user_window,
            "active_user",
        ] = 1
        df_days_frequency.loc[
            df_days_frequency["days_since_last_transaction"] > active_user_window,
            "active_user",
        ] = 0
        features_arr.append(df_days_frequency)

    new_df = join_dfs(groupby_cols, *features_arr)

    return new_df


def _calculate_frequency_features(
    df,
    groupby_cols,
    cols: Dict,
    time_col="month",
    freq="W",
):
    """Calculates frequency of orders per given time period.

    Args:
       df: the dataframe you want to calculate the features on
       groupby_cols: The columns to group by.
       transaction_id_col: The column that contains the transaction id.
       time_col: the column that contains the date of the transaction. Defaults to month
       freq: The frequency of the time column. Defaults to W

    Returns:
       A dataframe with the mean number of transactions per week for each customer.
    """
    transaction_id_col = itemgetter(
        "transaction_id",
    )(cols)

    freq_values = (
        df.groupby(groupby_cols + [time_col])
        .agg(
            mean_number_of_transactions_per_freq=pd.NamedAgg(
                transaction_id_col, "nunique"
            )
        )
        .reset_index()
        .fillna(0.0)
    ).rename(
        columns={
            "mean_number_of_transactions_per_freq": f"mean_number_of_transactions_per_{time_col}"
        }
    )

    base_df = df[groupby_cols].drop_duplicates()
    weeks = pd.DataFrame(
        pd.date_range(
            start=freq_values[time_col].min(),
            end=freq_values[time_col].max(),
            freq=freq,
        ),
        columns=[time_col],
    )

    base_df["tmp"] = 1
    weeks["tmp"] = 1
    base_df = base_df.merge(weeks, on=["tmp"], how="outer")

    freq_values["tmp"] = 1

    for col in groupby_cols + [time_col, "tmp"]:
        base_df[col] = base_df[col].astype(str)
        freq_values[col] = freq_values[col].astype(str)

    final_df = (
        freq_values.merge(base_df, on=groupby_cols + [time_col, "tmp"], how="outer")
        .fillna(0.0)
        .groupby(groupby_cols)
        .mean()
        .reset_index()
    )

    return final_df


def calculate_percentage_columns(df: pd.DataFrame, params: dict):
    """Calculates different percentage columns based on the parameters.

    Args:
      df (pd.DataFrame): the dataframe
      params (Dict): Dictionary with parameters.

    Returns:
      A dataframe with the added percentage features
    """
    final_df = df.copy()

    for col_name, feature_params in params.items():
        final_df = _calculate_percentage_of_feature_per_category(
            final_df, col_name, feature_params
        )

    return final_df


def _calculate_percentage_of_feature_per_category(
    df: pd.DataFrame,
    col_name: str,
    params: Dict,
):
    """It takes a dataframe, a column name, and a dictionary of parameters, and
    returns a new dataframe with the percentage of each feature in the
    percentage_cols list.

    Args:
      df (pd.DataFrame): the dataframe
      col_name (str): The name of the column that you want to add the percentage of.
      params (Dict): Dictionary with parameters.

    Returns:
      A dataframe with the percentage of each feature per category.
    """
    if "percentage_cols" in params:
        percentage_cols = params["percentage_cols"]
    elif "regex" in params:
        percentage_cols = list(df.filter(regex=params["regex"]).columns)

    main_col = params["main_col"]
    replace_name = params["replace_name"]
    sum_100 = params["sum_100"]

    new_df = df.copy()

    percentage_cols = [col for col in percentage_cols if col in new_df.columns]

    for col in percentage_cols:
        new_name = f"{col.replace(replace_name, col_name)}"
        if sum_100:
            new_df[new_name] = new_df[col] / new_df[percentage_cols].sum(axis=1)
        else:
            new_df[new_name] = new_df[col] / new_df[main_col]

    return new_df


def _pivot_and_rename(df: pd.DataFrame, groupby_cols: list) -> pd.DataFrame:
    """Pivot df and rename the columns.

    It pivots a dataframe and renames the columns
    Args:
      df (pd.DataFrame): the dataframe you want to pivot
      groupby_cols (list): The columns to group by.
    Returns:
      A dataframe with the columns of the original dataframe, but with the values of the original
    dataframe pivoted.
    """
    new_df = df.copy()

    if len(groupby_cols) != 1:
        new_df = new_df.pivot(groupby_cols[0], groupby_cols[1:])
        new_df.columns = [
            "_".join([str(col) for col in cols]) for cols in new_df.columns
        ]
    else:
        new_df.columns = groupby_cols + [
            col for col in new_df.columns if col not in groupby_cols
        ]

    new_df = new_df.reset_index()

    if "index" in new_df.columns:
        new_df.drop("index", axis=1, inplace=True)

    return new_df


def _nan_treatment_transactional(df: pd.DataFrame, verbose: Optional[bool] = False):
    """> For each column in the dataframe, treats the missing value based on
    the logic: If the column name contains any of the following strings:
    `days_since_first_transaction`, `days_between_transactions`, or
    `days_since_last_transaction`, then fill null values with the maximum value
    in the column. Otherwise, fill null values with 0.0.

    Args:
      df (pd.DataFrame): The dataframe to be treated
      verbose (Optional[bool]): If True, will print out the columns that are being treated. Defaults to False

    Returns:
      A dataframe with the null values filled in.
    """
    for col in df.columns:
        try:
            if (
                col.startswith("days_since_first_transaction")
                or "days_between_transactions" in col
                or "days_since_last_transaction" in col
            ):
                fill_value = df[col].max()
                if verbose:
                    logger.info(f"Filling null values for {col} with {fill_value}")
                df[col] = df[col].fillna(fill_value)
            else:
                if verbose:
                    logger.info(f"Filling null values for {col} with 0.0")
                df[col] = df[col].fillna(0.0)
        except:  # noqa E722
            logger.warn(f"Not able to treat col {col}")

    return df


def create_business_features(df, business_features_params):
    """Create business from catalog calls.

    It takes a dataframe and a list of dictionaries as input. Each dictionary in the list contains the
    name of a function and a dictionary of parameters to be passed to that function. The function then
    calls the function specified in the dictionary and passes the parameters to it

    Args:
      df: the dataframe to be transformed
      business_features_params: a list of dictionaries, each dictionary contains the following keys:

    Returns:
      A dataframe with the new features added.
    """
    if business_features_params is None:
        return df

    for params in business_features_params:
        foo = _load_obj(params["function"])
        df = foo(df, **params["parameters"])

    return df


def clean_transactions_prm(df: pd.DataFrame) -> pd.DataFrame:
    """> We drop the gross margin percentage column, group by the transaction
    id, customer id, sku, order delivered at date, sku macro category clean,
    and sku micro category, summing the remaining columns, and then merge the
    sku names back in.

    Args:
      df (pd.DataFrame): pd.DataFrame -> The dataframe to be cleaned

    Returns:
      A dataframe with the following columns:
        - transaction_id
        - customer_id
        - sku
        - order_delivered_at_date
        - sku_macro_category_clean
        - sku_micro_category
        - quantity
        - price
        - name_sku # add in sku_names if name exists
    """
    df = df.drop("gross_margin_percentage", axis=1)

    sku_names = df[["sku"]].groupby("sku").head(1).reset_index()

    df = (
        df.groupby(
            [
                "transaction_id",
                "customer_id",
                "sku",
                "order_delivered_at_date",
                "sku_macro_category_clean",
                "sku_micro_category",
            ]
        )
        .sum()
        .reset_index()
    )

    df = df.merge(sku_names, on=["sku"], how="left")

    logger.info(f"Number of rows: {df.shape[0]}")
    unique_clients = df["customer_id"].drop_duplicates().shape[0]
    logger.info(f"Number of unique_clients: {unique_clients}")

    return df


def _add_churn_column(df: pd.DataFrame, churn_col: str) -> pd.DataFrame:
    """It adds a column called `churn` to the dataframe, which is 1 if the user
    has churned and 0 otherwise.

    Args:
      df (pd.DataFrame): the dataframe to add the churn column to
      churn_col (str): the column name of the churn column in the dataframe

    Returns:
      A dataframe with a new column called churn.
    """
    logger.warning("Adding churn column")

    df[churn_col] = df[churn_col].apply(float)
    df["churn"] = df[churn_col].apply(lambda x: 1 if x <= 0 else 0)
    # fill as churned the users that have not made transactions
    df["churn"] = df["churn"].fillna(value=1)
    df["churn"] = df["churn"].apply(int)

    return df
