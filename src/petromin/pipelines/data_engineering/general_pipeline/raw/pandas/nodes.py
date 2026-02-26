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

import pandas as pd

from geospatial.geospatialde.cleaning import change_names, filter_nulls
from geospatial.geospatialde.validation import validate_columns
from segmentation_core.helpers.tag_managment.tag_dict import TagDict

logger = logging.getLogger(__name__)


def process_raw_pandas(data: pd.DataFrame, td: TagDict, source: str) -> pd.DataFrame:
    """Process origin into raw."""
    data = validate_columns(data, td, source)
    data = filter_nulls(data, td, source)
    # Uncomment this function for use: validate_dtypes(data, td, source)

    # Change names is last step where, we would need to modify tag dict to use the changed tag names
    data = change_names(data, td, source)

    return data
