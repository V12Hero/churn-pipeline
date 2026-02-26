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
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)
# from collections.abc import Iterable


def validate_no_duplicates(df: pd.DataFrame, join_cols: List[str] = ["customer_id"]):
    """
    It takes a dataframe and a column name, and checks if there are any duplicates in the dataframe

    Args:
      df (pd.DataFrame): the dataframe to validate
      join_col (str): the column to join on. Defaults to customer_id
    """
    n_duplicates = df.shape[0]

    n_no_duplicates = df[join_cols].drop_duplicates().shape[0]
    duplicates = n_duplicates - n_no_duplicates
    logger.info("Validating no duplicates")
    assert n_duplicates == n_no_duplicates, f"Found {(duplicates)} duplicates"


def check_duplicates(columns: List[str]):
    """A decorator that checks whether a function produces duplicate rows for a
    given column in the input DataFrame, and also checks whether the number of
    rows in the output DataFrame matches the number of rows in the input
    DataFrame.

    Args:
        columns (str): The name of the columns to check for duplicates.

    Returns:
        function: The decorated function.

    Raises:
        ValueError: If duplicates are found in the input DataFrame along the specified column, or if the number of
            rows in the output DataFrame does not match the number of rows in the input DataFrame.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            df = args[0]  # get the input DataFrame
            if df[columns].duplicated().any():
                raise ValueError(
                    f"Duplicates found in DataFrame along {columns} columns"
                )

            # check number of rows before and after applying the function
            before_rows = df.shape[0]
            func_output = func(*args, **kwargs)

            if isinstance(func_output, pd.DataFrame):
                output_df = func_output
            else:
                print(func_output)
                output_df = func_output[0]

            after_rows = output_df.shape[0]

            if before_rows != after_rows:
                raise ValueError(
                    f"Number of rows changed from {before_rows} to {after_rows} after applying function"
                )

            return func_output

        return wrapper

    return decorator
