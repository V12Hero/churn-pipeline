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

from kedro.pipeline import Pipeline, node, pipeline

from geospatial.geospatialde.cleaning import (
    drop_duplicates,
    map_values,
    pivot_values,
    remove_columns,
)
from geospatial.geospatialde.feature import aggregate_values, create_features


def create_pipeline(namespace="") -> Pipeline:
    """Create the pipelien for dataframes primary."""
    return pipeline(
        [
            node(
                func=create_features,
                inputs=["intermediate", "params:pre_features"],
                outputs="pre_features",
                name="create_pre_features",
            ),
            node(
                func=map_values,
                inputs=["pre_features", "td_updated", "params:source"],
                outputs="mapped_data",
                name="map_values",
            ),
            node(
                func=pivot_values,
                inputs=["mapped_data", "params:pivot"],
                outputs="pivoted_data",
                name="pivot_values",
            ),
            node(
                func=remove_columns,
                inputs=["pivoted_data", "td_updated", "params:source"],
                outputs="columns_removed",
                name="remove_columns",
            ),
            node(
                func=aggregate_values,
                inputs=["columns_removed", "params:groups", "params:aggregation"],
                outputs="aggregated_data",
                name="aggregate_values",
            ),
            node(
                func=create_features,
                inputs=["aggregated_data", "params:post_features"],
                outputs="features",
                name="create_post_features",
            ),
            node(
                func=drop_duplicates,
                inputs=["features"],
                outputs="primary",
                name="drop_duplicates",
            ),
        ],
        tags=["primary", f"primary_{namespace}"],
    )
