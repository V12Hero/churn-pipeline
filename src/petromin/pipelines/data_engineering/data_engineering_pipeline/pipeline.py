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

from kedro.pipeline import Pipeline, pipeline
from petromin.pipelines.data_engineering.general_pipeline import pipeline as general_pipeline
from petromin.pipelines.data_engineering.general_pipeline.raw.general import pipeline as raw_general
from petromin.pipelines.data_engineering.osm_pipeline import pipeline as osm_pipeline
from petromin.pipelines.data_engineering.spatial_mastertable_pipeline import pipeline as spatial_mastertable
from petromin.pipelines.data_engineering.users_pipeline import pipeline as users_pipeline

# from segmentation_core.helpers.parameters import load_parameters

# from .urbanicity_skyhook import pipeline as urbanicity_skyhook_pipeline

# params = load_parameters()


def create_pipeline() -> Pipeline:
    """All data engineering pipelines."""
    # general DE pipeline
    raw_general_pipe = raw_general.create_pipeline()
    users_pipe = users_pipeline.create_pipeline()

    users_de_pipe = pipeline(
        general_pipeline.create_pipeline(namespace="users"),
        inputs=["td"],
        parameters=["params:global_epsg"],
        namespace="users",
        tags=["users_de"],
    )

    countries = ["saudi_arabia"] #params["users"]["primary_location_grid"]["countries"]
    pipes_countries = []
    for country in countries:
        geospatial_pipe = pipeline(
            osm_pipeline.create_pipeline(),
            inputs=["td"],
            parameters=["params:global_epsg"],
            namespace=country,
            tags=[country],
        )

        spatial_mastertable_pipe = pipeline(
            spatial_mastertable.create_pipeline(country),
            inputs=["users.primary"],
            namespace=country,
        )

        pipes_countries += [geospatial_pipe + spatial_mastertable_pipe]

    pipelines = raw_general_pipe + users_pipe + users_de_pipe + sum(pipes_countries)

    return pipelines
