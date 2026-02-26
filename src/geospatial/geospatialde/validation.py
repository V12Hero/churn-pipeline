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

from segmentation_core.helpers.tag_managment.tag_dict import TagDict

logger = logging.getLogger(__name__)


def validate_columns(data: pd.DataFrame, td: TagDict, source: str) -> pd.DataFrame:
    """
    Validate that all columns in a given DataFrame are present in a specified tag
    dictionary and vice versa.
    """
    td.filter(condition={"source": source, "derived": False})
    tags = set(td.select())
    columns = set(data.columns)

    logger.info(columns)

    # Check that all columns of the df are in the dictionary
    if len(columns - tags) > 0:
        logging.warning(
            f"The following columns are not in the tag dictionary and will be filtered: {' ,'.join(columns-tags)}"
        )

    # Check that all columns of the dictionary are in the df
    if len(tags - columns) > 0:
        error = f"The following columns must be part of the dataframe: {' ,'.join(tags-columns)}"
        logging.error(error)
        raise KeyError(error)
    # return data
    logger.info(f"columns to be validated: {list(data.columns)}")
    logger.info(f"tags: {tags}")
    data = data[sorted(tags)]
    logger.info(f"validated columns: {list(data.columns)}")
    return data


def validate_dtypes(data: pd.DataFrame, td: TagDict, source: str) -> None:
    """
    Validate the data types of a pandas DataFrame against a specified TagDict and raises
    a TypeError if any mismatches are found.
    """
    td.filter(condition={"source": source, "derived": False})

    problems = [
        f"{col} is {dtype} and must be {td[col]['data_type']}"
        for col, dtype in data.dtypes.items()
        if dtype != td[col]["data_type"]
    ]

    if len(problems) > 0:
        logging.error(f"The following problems were found: {'/n'.join(problems)}")
        raise TypeError

    return None
