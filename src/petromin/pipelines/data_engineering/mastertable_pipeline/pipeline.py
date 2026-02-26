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

# from segmentation_core.helpers.parameters import load_parameters

from .nodes import join_pipelines_outputs

# params = load_parameters()
countries_list = ["KSA"] # params["join_pipeline_params"]["countries_list"]


def create_pipeline() -> Pipeline:
    """Extraction pipeline.

    Defines pipeline that extract data from all the raw systems/storages.
    All data is extracted to the intermediate layer, where names and types
    are adjusted.

    Returns:
        Kedro Pipeline object
    """
    pipe = pipeline(
        [
            node(
                func=join_pipelines_outputs,
                inputs={
                    "geospatial": "geospatial",
                    "params": "params:join_pipeline_params",
                    # "census": "censo",
                    "worldpop": "worldpop",
                    "transactions": "transactions",
                },
                outputs=["mastertable", "feature_store"],
                name="join_inputs",
                tags=["mastertable", "feature_store"],
            ),
        ],
        tags=["mastertable", "feature_store"],
    )
    pipes_countries = []
    for country in countries_list:
        mastertable_countries_pipe = pipeline(
            pipe,
            parameters=["params:join_pipeline_params"],
            tags=[country],
            namespace=country,
        )
        pipes_countries += [mastertable_countries_pipe]

    return sum(pipes_countries)
