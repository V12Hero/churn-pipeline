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
import multiprocessing
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tqdm

logger = logging.getLogger(__name__)


def join_dfs(join_col, *args):
    """
    Joins multiple datasets

    Args:
        join_col (str): Column to use on join.
        *args: List of datasets to join

    Returns:
        DataFrame
    """

    dfs_list = list(args)

    if type(join_col) == str:
        join_col = [join_col]
    for df in dfs_list:
        for col in join_col:
            df[col] = df[col].astype(str)

    df_final = dfs_list[0]
    for df in dfs_list[1:]:
        df_final = df_final.merge(df, on=join_col, how="left")

    return df_final


def parallelize_dataframe(df, func, n_cores=None):
    """
    It takes a dataframe, splits it into n_cores parts, and then runs a function on each part in
    parallel

    Args:
      df: The dataframe to be parallelized.
      func: The function to apply to each dataframe chunk.
      n_cores: The number of cores you want to use.

    Returns:
      A dataframe with the same columns as the original dataframe, but with the values of the columns
        replaced by the values returned by the function.
    """
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()
    logger.info(f"Using {n_cores} to parallelize process")
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(tqdm.tqdm(pool.imap(func, df_split), total=n_cores))
    pool.close()
    pool.join()

    return df


def _cast_id_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Casting id columns.

    Args:
        df (pd.DataFrame): data frame.
        col (str): col to cast as string.

    Returns:
        pd.DataFrame: cast id column.
    """
    df[col] = df[col].apply(int)
    df[col] = df[col].apply(str)
    return df


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def remove_accents_and_commas(input_str):
    return remove_accents(input_str).replace(",", "")


def replace_spaces_with_underscore(input_str):
    return input_str.replace(" ", "_")


def make_lowercase(input_str):
    return input_str.lower()


def clean_string(input_str):
    # Remove accents and commas
    cleaned_str = remove_accents_and_commas(input_str)

    # Replace spaces with underscores
    cleaned_str = replace_spaces_with_underscore(cleaned_str)

    # Make everything lowercase
    cleaned_str = make_lowercase(cleaned_str)

    return cleaned_str
