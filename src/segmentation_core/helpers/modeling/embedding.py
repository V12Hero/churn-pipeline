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
import warnings
from typing import Any, Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from .objects.load import load_object

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def embedding_sparse_features(df: pd.DataFrame, model_params: Dict) -> Dict[str, Any]:
    """Create embedded features from a dimensionality reduction technique.

    It takes a dataframe and a dictionary of model parameters, and returns a dictionary of dataframes,
    each of which contains the embedded features for a particular group of columns

    Args:
      df (pd.DataFrame): The dataframe that you want to embed
      model_params (Dict): A dictionary of dictionaries. The keys are the names of the embedding models.
    The values are dictionaries with the following keys:

    Returns:
      A dictionary of dictionaries.
    """
    return_dict: Dict[str, Any] = {}
    for name, params in model_params.items():
        col_group = params["columns_to_embed"]
        logger.info(f"Embedding columns for: {name}. Columns: {col_group}")
        return_dict[name] = {}
        return_dict[name]["col_group"] = col_group
        data = df[col_group]
        model = load_object(params["model"])
        scaler = load_object(params["scaler"])
        pipe = Pipeline([("scaler", scaler), ("embedding", model)])
        W = pipe.fit_transform(data)
        encoded_matrix = pd.DataFrame(
            W, columns=[f"component_{i}" for i in range(W.shape[1])]
        )
        try:
            variance_explained = round(
                pipe["embedding"].explained_variance_ratio_.sum(), 3
            )
            logger.info(f"Variance Explained: {variance_explained}")
            return_dict[name]["variance_explained"] = variance_explained
        except Exception as e:
            logger.info(e)
            return_dict[name]["variance_explained"] = "Unable to calculate"

        return_dict[name]["embedded_df"] = encoded_matrix
        return_dict[name]["embedding_pipeline"] = pipe
    return return_dict
