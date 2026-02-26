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

"""Transformers for outlier removal.
See
https://stackoverflow.com/questions/50928592/mypy-type-hint-unionfloat-int-is-there-a-number-type
for reason on float type hint use
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RangeOutlierRemover(BaseEstimator, TransformerMixin):
    """Remove outliers based on value range.

    Attributes:
        bounds: value range, values outside this range will be considered as outliers
        method: method used to handle outlier values
            Options:
                - `set_nan`: outliers are replaced by missing values
                - `drop`: rows with outliers are dropped from the data
                - `clip`: clip values outside the range
            Default: `set_nan`
    """

    def __init__(self, bounds: Tuple[float, float], method: str = "set_nan") -> None:
        """Instantiate a `RangeOutlierRemover` object."""
        self.bounds = bounds
        self.method = method
        self._higher_bound = None
        self._lower_bound = None

    @property
    def method(self):
        """Get method."""
        return self._method

    @method.setter
    def method(self, value):
        """Validate and set method."""
        if value not in ["set_nan", "drop", "clip"]:
            raise ValueError("Method must be one of `set_nan`, `drop` or `clip`")
        self._method = value

    @property
    def bounds(self):
        """Get bounds."""
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        """Validate and set bounds."""
        if not isinstance(value[0], float) or not isinstance(value[1], float):
            raise TypeError("Bounds values must be numerical")
        self._bounds = value

    def fit(
        self,
        X: pd.DataFrame,  # pylint: disable=unused-argument
        y: Optional[pd.Series] = None,  # pylint: disable=unused-argument
    ) -> RangeOutlierRemover:
        """Fit a `RangeOutlierRemover` object.

        Sets higher_bound and lower_bound values

        Args:
            X: unused, kept for compatibility
            y: unused, kept for compatibility

        Returns:
            Fitted `RangeOutlierRemover` object
        """
        self._higher_bound = max(self.bounds)
        self._lower_bound = min(self.bounds)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the input data.

        Uses one of the following methods:
            - `set_nan`: outliers are replaced by missing values
            - `drop`: rows with outliers are dropped from the data
            - `clip`: clip values outside the range

        Args:
            X: data to remove outliers from

        Returns:
            Data with outliers removed
        """
        if self.method == "clip":
            transformed_X = X.clip(
                lower=self._lower_bound, upper=self._higher_bound, axis=1
            )
        mask = X.gt(self._higher_bound) | X.lt(self._lower_bound)
        if self.method == "set_nan":
            transformed_X = X.mask(mask, other=np.nan)
        if self.method == "drop":
            transformed_X = X.loc[~mask.any(axis=1), :]
        return transformed_X


class QuantileRangeOutlierRemover(RangeOutlierRemover):
    """Remove outliers based on quantile range.

    Attributes:
        bounds: quantile range, values outside this quantile range will be considered
            as outliers
        method: method used to handle outlier values
            Options:
                - `set_nan`: outliers are replaced by missing values
                - `drop`: rows with outliers are dropped from the data
                - `clip`: clip values outside the range
            Default: `set_nan`
    """

    @property
    def bounds(self):
        """Get range."""
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        """Validate and set range."""
        if not isinstance(value[0], float) or not isinstance(value[1], float):
            raise TypeError("Bounds values must be numerical")
        if not 0 <= value[0] <= 1 or not 0 <= value[1] <= 1:
            raise ValueError("Bounds values should be between 0 and 1")
        self._bounds = value

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> QuantileRangeOutlierRemover:
        """Fit a `QuantileRangeOutlierRemover` object.

        Sets higher_bound and lower_bound values corresponding to the quantile range.

        Args:
            X: data used to compute the value range from the quantiles
            y: unused, kept for compatibility

        Returns:
            Fitted `QuantileRangeOutlierRemover` object
        """
        self._higher_bound = X.quantile(max(self.bounds))
        self._lower_bound = X.quantile(min(self.bounds))
        return self


class IQROutlierRemover(QuantileRangeOutlierRemover):
    """Remove outliers based on inter-quantile range multiplier.

    Attributes:
        bounds: quantile range
            Default: (0.25, 0.75)
        iqr_multiplier: inter-quantile range multiplier
            Default: 1.5
        method: method used to handle outlier values
            Options:
                - `set_nan`: outliers are replaced by missing values
                - `drop`: rows with outliers are dropped from the data
                - `clip`: clip values outside the range
            Default: `set_nan`
    """

    def __init__(
        self,
        bounds: Tuple[float, float] = (0.25, 0.75),
        iqr_multiplier: float = 1.5,
        method: str = "set_nan",
    ) -> None:
        """Instantiate an `IQROutlierRemover` object."""
        self.iqr_multiplier = iqr_multiplier
        super().__init__(bounds=bounds, method=method)

    @property
    def iqr_multiplier(self):
        """Get IQR multiplier."""
        return self._iqr_multiplier

    @iqr_multiplier.setter
    def iqr_multiplier(self, value):
        """Validate and set IQR multiplier."""
        if not isinstance(value, float):
            raise TypeError("IQR multiplier must be numerical")
        self._iqr_multiplier = value

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> IQROutlierRemover:
        """Fit an `IQROutlierRemover` object.

        Sets higher_bound and lower_bound values corresponding to the quantile range.

        Args:
            X: data used to compute the value range from the quantiles
            y: unused, kept for compatibility

        Returns:
            Fitted `IQROutlierRemover` object
        """
        lqv, uqv = X.quantile(self.bounds[0]), X.quantile(self.bounds[1])
        self._higher_bound = uqv + (uqv - lqv) * self.iqr_multiplier
        self._lower_bound = lqv - (uqv - lqv) * self.iqr_multiplier
        return self
