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

"""Central Tag Management class."""

import logging
from typing import Any, Dict, List, TypeGuard, Union

import pandas as pd

from .dependencies import DependencyGraph
from .validation import TagDictError, validate_td

Condition = Union[list, str, float, int]
ConditionDict = Dict[str, Condition]


def is_condition(c: Any) -> TypeGuard[Condition]:
    """`is_condition` returns `True` if `c` is a list, string, float, or
    integer.

    Args:
      c (Any): Any

    Returns:
      A type guard.
    """
    return isinstance(c, (list, str, float, int))


def is_condition_dict(D: Any) -> TypeGuard[ConditionDict]:
    """`is_condition_dict` returns `True` if all the values in the dictionary
    are conditions.

    Args:
      D (Any): Any

    Returns:
      A type guard that checks if the input is a condition dictionary.
    """
    return all([is_condition(v) for _, v in D.items()])


logger = logging.getLogger(__name__)


class TagDict:
    """Class to hold data dictionary.

    Class to hold a data dictionary. Uses a dataframe underneath and
    takes care of QA and convenience methods.
    """

    def __init__(self, data: pd.DataFrame, validate: bool = True):
        """Default constructor.

        Creates new TagDict object from pandas dataframe.

        Args:
            data: input dataframe
            validate: whether to validate the input dataframe. validate=False can
                lead to a dysfunctional TagDict but may be useful for testing
        """
        self._validate = validate
        self._data = validate_td(data) if self._validate else data

        self._update_dependency_graph()

    def _update_dependency_graph(self):
        """
        Update dependency graph to reflect what is currently in the tag
        dict.
        """
        graph = DependencyGraph()
        if "on_off_dependencies" in self._data.columns:
            all_deps = self._data.set_index("tag")["on_off_dependencies"].dropna()
            for tag, on_off_dependencies in all_deps.items():
                for dep in on_off_dependencies:
                    graph.add_dependency(tag, dep)
        self._dep_graph = graph

    def to_frame(self) -> pd.DataFrame:
        """Transform tagdict as dataframe.

        Returns:
            underlying dataframe
        """
        data = self._data.copy()
        data["on_off_dependencies"] = data["on_off_dependencies"].apply(", ".join)
        return data

    def _check_key(self, key: str):
        """Check if a key is a known tag."""
        if key not in self._data["tag"].values:
            raise KeyError(f"Tag `{key}` not found in tag dictionary.")

    def _check_is_on_off(self, key: str):
        """Check if a key is an on_off type tag."""
        tag_data = self[key]
        if tag_data["tag_type"] != "on_off":
            raise TagDictError(
                f"Tag `{key}` is not labelled as 'on_off' tag_type in the tag dictionary."
            )

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Enable subsetting by tag to get all information about a given tag.

        Args:
            key: tag name
        Returns:
            dict of tag information
        """
        self._check_key(key)
        data = self._data
        dictionary = data.loc[data["tag"] == key, :].iloc[0, :].to_dict()
        # Change NaN for None
        return {k: (None if pd.isnull(v) else v) for k, v in dictionary.items()}

    def __contains__(self, key: str) -> bool:
        """Checks whether a given tag has a tag dict entry.

        Args:
            key: tag name
        Returns:
            True if tag in tag dict.
        """
        return key in self._data["tag"].values

    def name(self, key: str) -> str:
        """Returns clear name for given tag if set or tag name if not.

        Args:
            key: tag name
        Returns:
            clear name
        """
        tag_data = self[key]
        return tag_data["name"] or key

    def dependencies(self, key: str) -> List[str]:
        """Get all on_off_dependencies of a given tag.

        Args:
            key: input tag
        Returns:
            list of tags that depend on input tag
        """
        self._check_key(key)
        return self._dep_graph.get_dependencies(key)

    def dependents(self, key: str) -> List[str]:
        """Get all dependents of a given tag.

        Args:
            key: input tag
        Returns:
            list of tags that input tag depends on
        """
        self._check_key(key)
        self._check_is_on_off(key)
        return self._dep_graph.get_dependents(key)

    def add_tag(self, tag_row: Union[dict, pd.DataFrame]):
        """Adds new tag row/s to the TagDict instance.

        Only if and entry doesn't already exist.

        Args:
            tag_row: DataFrame or Series/dict-like object of tag row/s
        Raises:
            TagDictError if the supplied tag rows are incorrect
        """
        if not isinstance(tag_row, (dict, pd.DataFrame)):
            raise TagDictError(
                f"Must provide a valid DataFrame or "
                f"dict-like object for the tag row/s. Invalid "
                f"object of type {type(tag_row)} provided"
            )
        # Skip tags if already present in the TagDict.
        tag_data = pd.DataFrame(data=tag_row)
        tag_data.set_index("tag", inplace=True)

        tags_already_present = set(tag_data.index).intersection(set(self._data["tag"]))
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

            self._data = validate_td(data) if self._validate else data

            self._update_dependency_graph()

    def select(self, filter_col: str = None, condition: Any = None) -> List[str]:
        """Retrieves all tags according to a given column and condition.

        If no filter_color condition is given then all tags are returned.

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
        """

        def _condition(x: Any):
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
                raise KeyError(f"Column `{filter_col}` not found.")

            mask = data[filter_col].apply(_condition) > 0

        else:
            mask = data.apply(_condition, axis=1) > 0

        return list(data.loc[mask, "tag"])

    def _check_col(self, col: str) -> None:
        """Check that the column is in the dataframe."""
        if col not in self._data.columns:
            logging.error(f"Column {col} not found in feature dictionary")

    def _filter_col(self, filter_col: str, condition: Condition):
        """Filter the underlying dataframe based on a condition."""
        self._check_col(filter_col)

        if isinstance(condition, list):
            self._data = self._data.loc[self._data[filter_col].isin(condition)]
        elif isinstance(condition, (str, float, int)):
            self._data = self._data.loc[self._data[filter_col] == condition]
        else:
            logging.error("Expected condition list, str, float or int")

    def filter(
        self,
        filter_col: str = None,
        condition: Union[ConditionDict, Condition] = True,
    ) -> None:
        """
        Filter the underlying dataframe
        Parameters
        ----------
        filter_col: str
            Column be used for filtering. If it is not specied the function
            will expect a dictionary as condition.
        condition: dict, list, str, float, int
            - A list of values to perform .isin()
            - A dictionary with the keys as the column name and the values as the
            condition that the column needs to satisfy. If you want to use this
            option, set filter_col as None
            - A str, float, int or bool with the condition for the column specified
            in filter_col
        """
        if is_condition(condition):
            self._filter_col(filter_col, condition)
        elif is_condition_dict(condition):
            for col, filter in condition.items():
                self._filter_col(col, filter)
        elif (filter_col is not None) and isinstance(condition, dict):
            raise TypeError("Cannot especify filter_col and condition as dictionary")

        if len(self._data) == 0:
            raise Exception(
                f"This filter returned no results: {filter_col}=={condition}"
            )

    def change_tag(self, tag: str):
        """Change the current tag for another column."""
        data = self._data.copy()
        data["tag"] = data[tag]

        self._data = validate_td(data) if self._validate else data

        self._update_dependency_graph()

    def keys(self):
        """Get the column names of the data."""
        return list(self._data.columns)
