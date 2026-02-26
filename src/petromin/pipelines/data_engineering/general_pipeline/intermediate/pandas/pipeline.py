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

from petromin.pipelines.data_engineering.general_pipeline.intermediate.pandas.nodes import tag_dict_preprocessing
from kedro.pipeline import Pipeline, node, pipeline

from geospatial.geospatialde.cleaning import (
    change_epsg,
    standarize_string_formats_dataframes,
)
from geospatial.geospatialde.feature import create_geometry


def create_pipeline(namespace="") -> Pipeline:
    """Create the intermediate pandas pipeline."""
    return pipeline(
        [
            node(
                func=tag_dict_preprocessing,
                inputs=["raw", "td_updated", "params:source"],
                outputs="td_preprocess",
                name="tag_dict_based_preprocessing",
            ),
            node(
                func=create_geometry,
                inputs=[
                    "td_preprocess",
                    "params:apply",
                    "params:coordinates",
                    "params:epsg",
                ],
                outputs="geometry",
                name="create_geometry",
            ),
            node(
                func=change_epsg,
                inputs=["geometry", "params:global_epsg"],
                outputs="df_change_epsg",
                name="change_epsg",
            ),
            node(
                func=standarize_string_formats_dataframes,
                inputs=["df_change_epsg", "params:cols_to_standarize"],
                outputs="intermediate",
                name="standarize_string_formats_dataframes",
            ),
        ],
        tags=["intermediate", f"intermediate_{namespace}"],
    )
