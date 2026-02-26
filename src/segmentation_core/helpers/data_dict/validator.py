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

"""Contains functions that support the validation of the data dictionary."""

from typing import Dict, List, Set, Tuple

import pandas as pd

SUPPORTED_DATA_TYPES = {
    "numeric": ["int", "float"],
    "categorical": ["str"],
    "datetime": ["date", "timestamp"],
    "boolean": ["bool"],
}
REQUIRED_COLUMNS = {"column_id", "feature_group", "business_name", "data_type"}
UNIQUE = {"column_id", "business_name"}
KNOWN_VALUES = {"data_type": {"int", "float", "bool", "str", "date", "timestamp"}}
# column_ids / featyres are checked for whether they break
# any of the below rules captured as rule - explanation
ILLEGAL_FEATURE_PATTERNS = [
    (r"^.*,+.*$", "no commas in column_id"),
    (r"^\s.*$", "tag must not start with whitespace character"),
    (r"^.*\s$", "tag must not end with whitespace character"),
]


class DataDictError(Exception):
    """Tag Dictionary related exceptions."""


def check_missing_cols(required_cols: Set[str], data: pd.DataFrame) -> None:
    """Check if a list of required columns are in a dataframe.

    Args:
          required_cols: List of required columns to check
          data: The dataframe to be checked

    Raises:
        DataDictError: If there are missing columns
    """
    missing_cols = set(required_cols) - set(data.columns)
    if missing_cols:
        raise DataDictError(
            "The following columns are missing from the input dataframe: {}".format(
                missing_cols
            )
        )


def check_duplicate_vals(cols_to_check: Set[str], data: pd.DataFrame) -> None:
    """Check if no duplicated columns in the DataFrame.

    Checks if there are duplicate values in a list of specified columns in the provided
    DataFrame

    Args:
        cols_to_check: List of required columns to check
        data: The dataframe to be checked

    Raises:
        DataDictError: If any of the specified columns have duplicate values
    """
    for col in cols_to_check:
        duplicates = data.loc[data[col].duplicated(), col]
        if not duplicates.empty:
            raise DataDictError(
                "The following values are duplicated in column `{}`: {}".format(
                    col, list(duplicates)
                )
            )


def check_illegal_records(
    invalid_regex_patterns: List[Tuple[str, str]],
    data: pd.DataFrame,
    column: str = "column_id",
) -> None:
    """Check if column records follow the expected format.

    Checks if there are records in a column that do not conform to expected format
    in the provided dataframe.

    Args:
        invalid_regex_patterns: This is a tuple containing the regex pattern and
            corresponding rule e.g. (r"^.*,+.*$", "no commas in column name")
        data: The dataframe to be checked
        column: The column to be checked. By default, this is the column_id

    Raises:
        DataDictError: If any of the specified column has an illegal record
    """
    for pattern, rule in invalid_regex_patterns:
        matches = data.loc[data[column].str.match(pattern), column]
        if not matches.empty:
            raise DataDictError(
                "The following column_id don't adhere to rule `{}`: {}".format(
                    rule, list(matches)
                )
            )


def check_known_values(known_values: Dict[str, Set[str]], data: pd.DataFrame) -> None:
    """Check if there are certain values in a column that are NOT expected.

    Args:
        known_values: A mapping with column as key and values as a set of expected
            values for the column
        data: The dataframe to be checked

    Raises:
        DataDictError: If there are invalid values in a the list of columns specified
    """
    for col, known_vals in known_values.items():
        invalid = set(data[col].str.lower().dropna()) - known_vals
        if invalid:
            raise DataDictError(
                "Found invalid entries in column {}: {}. Must be one of: {}".format(
                    col, invalid, known_vals
                )
            )


def validate_dict(use_case_id: str, data: pd.DataFrame) -> pd.DataFrame:
    """Run check on data dictionary provided by the user.

    Args:
        use_case_id: The identifier of the use case
            for current data dictionary
        data: tag dict data frame

    Returns:
        validated dataframe with comma separated values parsed to lists

    Raises:
        DataDictError: If any of the checks fail
    """
    data = data.copy()
    use_case_cols = {
        f"{use_case_id}:feature",
        f"{use_case_id}:target",
        f"{use_case_id}:key",
    }
    all_required_cols = set(REQUIRED_COLUMNS).union(set(use_case_cols))

    check_missing_cols(all_required_cols, data)
    check_duplicate_vals(UNIQUE, data)
    check_illegal_records(ILLEGAL_FEATURE_PATTERNS, data)
    check_known_values(KNOWN_VALUES, data)

    # validate markers of key, features, target variable
    allowed_marker_values = ["Y", "N", None]
    marker_values_found = []

    for col in use_case_cols:
        marker_values_found.extend(list(data[col].unique()))

    if len(set(marker_values_found) - set(allowed_marker_values)) > 0:
        raise DataDictError("markers can only be {'Y', 'N', None}")

    # Check number of features, target, and key
    if len(data[data[f"{use_case_id}:feature"] == "Y"]) < 2:
        raise DataDictError(
            f"There should be at least 2 features "
            f"specified for use_case:{use_case_id}"
        )

    if len(data[data[f"{use_case_id}:target"] == "Y"]) != 1:
        raise DataDictError(
            f"There should only be 1 column_id "
            f"representing target variable for use_case:{use_case_id}"
        )

    if len(data[data[f"{use_case_id}:key"] == "Y"]) != 1:
        raise DataDictError(
            f"There should only be 1 column_id "
            f"representing key for use_case:{use_case_id}"
        )

    return data
