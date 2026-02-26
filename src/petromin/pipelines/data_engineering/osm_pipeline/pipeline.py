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

import geopandas as gpd
import numpy as np
from petromin.pipelines.data_engineering.general_pipeline.intermediate.general import pipeline as general_int
from petromin.pipelines.data_engineering.general_pipeline.intermediate.pandas import pipeline as intermediate
from petromin.pipelines.data_engineering.general_pipeline.primary.pandas import pipeline as primary
from petromin.pipelines.data_engineering.general_pipeline.raw.pandas.nodes import process_raw_pandas
from kedro.pipeline import Pipeline, node, pipeline

from segmentation_core.helpers.tag_managment.tag_dict import TagDict

from .osmosis_nodes import osmosis_preprocessing


def create_pipeline() -> Pipeline:
    """Create the geospatial pipeline for each country."""
    # Define template with preprocessing dependency
    namespace = "geospatial"

    def process_raw_pandas_osmosis(
        origin: gpd.GeoDataFrame, osmosis_success: bool, td: TagDict, source: str
    ):
        assert osmosis_success

        td.filter(condition={"source": source, "derived": False})
        tags = set(td.select())
        columns = set(origin.columns)

        missing_cols = list(tags - columns)
        origin[missing_cols] = np.nan

        raw = process_raw_pandas(origin, td, source)

        return raw

    general_pipe = pipeline(
        [
            node(
                func=process_raw_pandas_osmosis,
                inputs=["origin", "osmosis_success", "td", "params:source"],
                outputs="raw",
                name="process_raw",
            ),
            general_int.create_pipeline(namespace),
            intermediate.create_pipeline(namespace),
            primary.create_pipeline(namespace),
        ],
    )

    # Create instances of templates
    general_pipes = ["osm_subway", "osm_poi", "osm_highways"]
    general_pipes = [
        pipeline(
            general_pipe,
            inputs=["td", "osmosis_success"],
            parameters=["params:global_epsg"],
            tags=[namespace],
            namespace=namespace,
        )
        for namespace in general_pipes
    ]

    # Merged pipeline with preprocessing node
    pipe = pipeline(
        [
            node(
                func=osmosis_preprocessing,
                inputs=["params:bbox", "params:osm_filepath", "params:country_name"],
                outputs=["osmosis_success"],
                tags=["osmosis"],
            ),
            *general_pipes,
        ],
        tags=["geospatial_de", "osm_de"],
    )
    return pipe
