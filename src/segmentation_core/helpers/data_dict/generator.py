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

"""This module creates a data dictionary object from file.

This module creates a data dictionary object from file and includes supporting utility
functions for.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import validator

logger = logging.getLogger(__name__)


class DataDict:
    """Class to hold a data dictionary.

    Uses a dataframe underneath and takes care of QA and convenience methods.
    """

    def __init__(self, use_case_name: str, data: pd.DataFrame, validate: bool = True):
        """Create new TagDict object from pandas dataframe.

        Args:
            use_case_name: The identifier of the use case
            data: input dataframe
            validate: whether to validate the input dataframe. validate=False can
                lead to a dysfunctional TagDict but may be useful for testing
        """
        self._validate = validate
        self._use_case = use_case_name
        self._data = validator.validate_dict(use_case_name, data) if validate else data

    @property
    def use_case(self):
        """Return the use case id for the current data dictionary."""
        return self._use_case

    def get_features(self, data_type: str = None) -> List[str]:
        """Get features.

        Args:
            data_type: This can be set the following:
                - numeric: This will get all the features with data_type in
                    numeric_types = ["int", "float"]
                - boolean: This will get all the features with data type "bool"
                - categorical: This will get all the features with
                    data_type in categorical_types = ["str"]
                - datetime: This will get all the features with
                    data_type in datetime_types
                If not specified, the functions returns all features
                    for given use case

        Returns:
            list of column_id representing features

        Raises:
            NotImplementedError: This is raised when provided data_type
                is not in {"numeric", "categorical", "datetime"}
        """
        use_case_id = self._use_case
        feature_column = f"{use_case_id}:feature"
        data = self._data.copy()

        feature_rows = data[data[feature_column] == "Y"]

        if data_type:
            if data_type not in validator.SUPPORTED_DATA_TYPES:
                raise NotImplementedError(f"{data_type} is NOT supported")

            data_types = validator.SUPPORTED_DATA_TYPES[data_type]
            feature_rows = feature_rows[
                feature_rows["data_type"].str.lower().isin(data_types)
            ]

        return feature_rows["column_id"].to_list()

    def _get_variable(self, name: str) -> Optional[str]:
        """Get the name of the variable as defined in the data dictionary.

        Args:
            name: variable name

        Returns:
            column_id of the variable

        Raises:
            KeyError: if key not present
            IndexError: if no row is available for the selected key
        """
        use_case_id = self._use_case
        col_name = f"{use_case_id}:{name}"
        data = self._data.copy()

        try:
            target_rows = data[data[col_name] == "Y"]

            return target_rows["column_id"].values[0]
        except (KeyError, IndexError):
            return None

    def get_target_variable(self) -> Optional[str]:
        """Get the name of the target variable as defined in the data dictionary.

        Returns:
            column_id of the target_variable
        """
        return self._get_variable("target")

    def get_target_switch_variable(self) -> Optional[str]:
        """Get the name of the column that flags observations.

        Get the name of the column that flags observations that have switched during
        the hold out period.

        Returns:
            column_id of the target_switch
        """
        return self._get_variable("target_switch")

    def get_date_column(self) -> Optional[str]:
        """Get the name of the date column as defined in the data dictionary.

        Returns:
            column_id of the date column
        """
        return self._get_variable("date")

    def get_key_column(self) -> Optional[str]:
        """Get the name of the key column as defined in the data dictionary.

        Returns:
            column_id of the key column
        """
        return self._get_variable("key")

    @classmethod
    def from_dict(cls, use_case_name: str, data: Dict, validate: bool = True, **kwargs):
        """Alternative constructor.

        Creates new TagDict object from a dictionary. The dict should be structured in
        what pandas calls "index orientation", i.e.
        `{"feat1": {"feature_group": xxx, "business_name": xx, ..},
          "feat2": {"feature_group": ...}`

        Args:
            use_case_name: The identifier of the use case
            data: input dict
            validate: whether to validate the input dict. validate=False can
             lead to a dysfunctional TagDict but may be useful for testing
            **kwargs: additional keyword arguments for pandas.DataFrame.from_dict()

        Returns:
            DataDict: data dictionary object
        """
        df = pd.DataFrame.from_dict(data, orient="index", **kwargs)
        df.index.name = "column_id"

        return cls(use_case_name, df.reset_index(), validate)

    @classmethod
    def from_json(cls, use_case_name: str, data: str, validate: bool = True, **kwargs):
        """Alternative constructor.

        Creates new TagDict object from a json string. The json object should be
        structured in what pandas calls "index orientation", i.e.
        `{"feat1": {"feature_group": xxx, "business_name": xx, ..},
          "feat2": {"feature_group": ...}`

        Args:
            use_case_name: The identifier of the use case
            data: json string
            validate: whether to validate the input str. validate=False can
                      lead to a dysfunctional TagDict but may be useful for testing
            **kwargs: additional keyword arguments for pandas.read_json()

        Returns:
            DataDict: data dictionary object
        """
        df = pd.read_json(data, orient="index", **kwargs)
        df.index.name = "column_id"

        return cls(use_case_name, df.reset_index(), validate)

    def to_frame(self) -> pd.DataFrame:
        """Get underlying dataframe.

        Returns:
            underlying dataframe
        """
        data = self._data.copy()
        return data

    def to_dict(self) -> Dict:
        """Get  dictionary representation of the underlying dataframe.

        Returns the dictionary representation of the underlying dataframe in
        what pandas calls "index orientation", i.e.
        `{"feat1": {"feature_group": xxx, "business_name": xx, ..},
          "feat2": {"feature_group": ...}`

        Returns:
            Dict representation of underlying dataframe.
        """
        df = self.to_frame().set_index("column_id")
        return df.to_dict(orient="index")

    def to_json(self, **kwargs) -> str:
        """Get JSON string representation of the underlying dataframe.

        Returns the JSON string representation of the underlying dataframe in
        what pandas calls "index orientation", i.e.
        `{"feat1": {"feature_group": xxx, "business_name": xx, ..},
          "feat2": {"feature_group": ...}`

        Args:
            **kwargs: additional keyword arguments for pandas.DataFrame.to_json()

        Returns:
            json string representation of underlying dataframe.
        """
        df = self.to_frame().set_index("column_id")
        return df.to_json(orient="index", **kwargs)

    def _check_key(self, key: str):
        """Check if a key is a known tag.

        Args:
            key:

        Raises:
            KeyError: This is raised when a specified column_id does not exists in the
                data dictionary
        """
        if key not in self._data["column_id"].values:
            raise KeyError("column_id `{}` not found in tag dictionary.".format(key))

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Enable subsetting by tag to get all information about a given tag.

        Args:
            key: tag name

        Returns:
            dict of tag information
        """
        self._check_key(key)
        data = self._data
        return data.loc[data["column_id"] == key, :].iloc[0, :].to_dict()

    def __contains__(self, key: str) -> bool:
        """Check whether a given tag has a tag dict entry.

        Args:
            key: tag name

        Returns:
            True if tag in tag dict.
        """
        return key in self._data["column_id"].values

    def name(self, key: str) -> str:
        """Return clear name for given tag if set or tag name if not.

        Args:
            key: tag name

        Returns:
            clear name
        """
        tag_data = self[key]
        return tag_data["business_name"] or key

    def add_tag(self, feature_row: Union[dict, pd.DataFrame]):
        """Add new tag to the TagDict instance.

        Adds new tag row/s to the TagDict instance, only if and entry doesn't already
        exist.

        Args:
            feature_row: DataFrame or Series/dict-like object of tag row/s

        Raises:
            DataDictError: if the supplied tag rows are incorrect
        """
        if not isinstance(feature_row, (dict, pd.DataFrame)):
            raise validator.DataDictError(
                f"Must provide a valid DataFrame or "
                f"dict-like object for the tag row/s. Invalid "
                f"object of type {type(feature_row)} provided"
            )
        # Skip tags if already present in the TagDict.
        tag_data = pd.DataFrame(data=feature_row)
        tag_data.set_index("column_id", inplace=True)

        tags_already_present = set(tag_data.index).intersection(
            set(self._data["column_id"])
        )
        if tags_already_present:
            logger.info(
                f"[{tags_already_present}] already present in the Tag "
                f"Dictionary. Skipping."
            )
            tag_data.drop(list(tags_already_present), inplace=True)

        if not tag_data.empty:
            data = self.to_frame()
            tag_data.reset_index(inplace=True)
            data = data.append(tag_data, ignore_index=True, sort=False)

            self._data = data

    def select(self, filter_col: str = None, condition: Any = None) -> List[str]:
        """Get all tags base on given column and condition.

        Retrieves all tags according to a given column and condition. If no filter_col
        or condition is given then all tags are returned.

        Args:
            filter_col: optional name of column to filter by
            condition: filter condition
                       if None: returns all tags where filter_col > 0
                       if value: returns all tags where filter_col == values
                       if callable: returns all tags where filter_col.apply(callable)
                       evaluates to True if filter_col is present, or
                       row.apply(callable) evaluates to True if filter_col
                       is not present

        Returns:
            list of tags

        Raises:
            TypeError:
            KeyError: This is raised when a specified column_id is not found
        """

        def _condition(x):
            # handle case where we are given a callable condition
            if callable(condition):
                return condition(x)

            # if condition is not callable, we will assert equality
            if condition:
                return x == condition

            # check if x is iterable (ie a row) or not (ie a column)
            try:
                iter(x)
            except TypeError:
                # x is a column, check > 0
                return x > 0 if x else False

            # x is a row and no condition is given, so we return
            # everything (empty select)
            return True

        data = self._data

        if filter_col:
            if filter_col not in data.columns:
                raise KeyError("Column `{}` not found.".format(filter_col))

            mask = data[filter_col].apply(_condition) > 0

        else:
            mask = data.apply(_condition, axis=1) > 0

        return list(data.loc[mask, "column_id"])


def create_data_dict(data: pd.DataFrame, params: dict) -> DataDict:
    """Create a data dictionary.

    Args:
        data: The dataframe containing the data dictionary
        params: List of parameters relevant to the use case

    Returns:
        data_dict: An instantiated DataDict object
    """
    data_dict = DataDict(params["use_case_id"], data)
    return data_dict
