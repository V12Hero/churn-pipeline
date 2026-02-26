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

from petromin.pipelines.data_engineering.general_pipeline import pipeline as general_pipeline
from kedro.pipeline import Pipeline, node, pipeline

from transactions.nodes import clean_transactions_prm, prepare_transactional_data

from .feature import pipeline as feature

logger = logging.getLogger(__name__)


def create_pipeline() -> Pipeline:
    """Create the transactions pipeline."""
    pipeline_list = pipeline(
        [
            pipeline(
                general_pipeline.create_pipeline("transactions"),
                outputs={"primary": "primary_general"},
            ),
            node(
                func=clean_transactions_prm,
                inputs="primary_general",
                outputs="primary",
                tags=["clean_trx"],
            ),
            node(
                prepare_transactional_data,
                inputs=["primary", "params:global_params"],
                outputs="transactions_with_evaluation_date",
                tags=[
                    "transactions",
                    "add_days_since",
                ],
            ),
        ],
        inputs=["td"],
        parameters=["params:global_epsg"],
        tags=["transactions_de"],
        namespace="transactions",
    )

    pipeline_list += feature.create_pipeline("transactions_120d")
    pipeline_list += feature.create_pipeline("transactions_90d")

    return pipeline_list
