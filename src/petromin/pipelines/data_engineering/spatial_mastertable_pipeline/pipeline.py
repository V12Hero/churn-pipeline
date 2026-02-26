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

from geospatial.geospatialde.cleaning import filter_country
from geospatial.geospatialde.feature import (
    combine_geospatial_features,
    compute_geographic_features,
    compute_haversine_distance,
)
from segmentation_core.helpers.data_processing.general import join_dfs
# from segmentation_core.helpers.parameters import load_parameters


def create_geographic_distance_pipeline(primaries) -> Pipeline:
    """
    1. Compute haversined distance within a specified radius, in parallel for each city in the bbox dict.
    2. Compute # of POI in list of radiuses (must be less than max_diameter), and distance

    Improvements:
    The distance is the max radius if no POI found
    """

    def create_nodes(source):
        nodes = [
            node(
                func=compute_haversine_distance,
                inputs=[
                    "primary",
                    f"{source}.primary",
                    f"params:feature.osm_tags.{source}",
                    "params:bbox",
                ],
                outputs=f"haversine_distance_{source}",
                name=f"compute_distance_{source}",
            ),
            node(
                func=compute_geographic_features,
                inputs=[
                    f"haversine_distance_{source}",
                    "primary",
                    f"params:feature.osm_tags.{source}.thresholds",
                    f"params:feature.osm_tags.{source}.max_diameter",
                    f"params:feature.osm_tags.{source}.columns",
                    "params:feature.groupby_col",
                ],
                outputs=f"{source}.feature",
                name=f"{source}_geographic_features",
            ),
        ]

        return pipeline(nodes)

    return sum([create_nodes(primary) for primary in primaries])


def create_pipeline(country, osm_tags=[]) -> Pipeline:
    """Create the pipeline for the geospatial master table."""

    osm_tags_inputs = []
    osm_tags_feature = []
    if len(osm_tags) <= 0:
        # params = load_parameters()
        osm_tags =  ["osm_subway", "osm_poi", "osm_highways"]
        osm_tags_inputs = [f"{tag}.primary" for tag in osm_tags]
        osm_tags_feature = [f"{tag}.feature" for tag in osm_tags]

    return pipeline(
        [
            node(
                func=filter_country,
                inputs=["users.primary", "params:country"],
                outputs="primary",
                name="filter_country",
                tags=["country_primary"],
            ),
            create_geographic_distance_pipeline(osm_tags),
            # Joining all the datasets
            node(
                join_dfs,
                inputs=["params:feature.groupby_col", "primary", *osm_tags_feature],
                outputs="joined.feature",
                tags="join_users",
            ),
            node(
                combine_geospatial_features,
                inputs=["joined.feature", "params:feature.combine_geospatial_features"],
                outputs="raw_master_table",
                tags="join_users",
            ),
        ],
        inputs=[
            "users.primary",
            *osm_tags_inputs,
        ],
        outputs=["primary"],
        tags=["feature", "users", "users_feature", "users_de", "spatial_mastertable"],
        namespace="spatial_mastertable",
    )
