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

"""Contains functions for running customerone_commons data transformation tasks.

This module contains functions for running data transformation
tasks including: imputation, label encoding, and outlier removal
"""

from segmentation_core.helpers.data_transformers.cleaning_utils import (  # noqa
    _apply_type,
    _convert_to_float,
    _deduplicate_pandas_df_columns,
    _df_values_type,
    _drop_col_if_present,
    _drop_column_with_threshold_of_nans,
    _get_type,
    _replace_elements,
    _select_cols,
    _standarize_column_names,
    _unidecode_strings,
    apply_outlier_remove_rule,
    convert_bool,
    deduplicate_pandas,
    enforce_custom_schema,
    filling_nans_by_fixed_value,
    remove_cols,
    series_convert_bool,
)
from segmentation_core.helpers.data_transformers.outlier_removal import (  # noqa
    IQROutlierRemover,
    QuantileRangeOutlierRemover,
    RangeOutlierRemover,
)
from segmentation_core.helpers.data_transformers.transformer_utils import (  # noqa
    deduplicate_df_using_dict,
    enforce_schema_using_dict,
    get_cols_to_skip,
    get_features_to_impute,
    get_model_input_datasets,
)
from segmentation_core.helpers.data_transformers.transformers import (  # noqa
    AbstractMLTransformer,
    SelectColumnTransformer,
    apply_fitted_transformer,
    apply_fitted_transformer_node,
    apply_transformer_multiple_sets,
    fit_transformer,
)

from . import cleaning_utils, outlier_removal

__all__ = ["cleaning_utils", "outlier_removal"]
