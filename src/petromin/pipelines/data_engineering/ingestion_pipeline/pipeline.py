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

from .nodes import (
    ingest_branches,
    ingest_customers,
    ingest_invoices,
    ingest_invoices_items_PE,
    ingest_invoices_items_PAC,
    ingest_invoices_items,
    ingest_promos,
    ingest_transactions,
    ingest_vehicles
)


def create_pipeline() -> Pipeline:
    """Extraction pipeline.

    Defines pipeline that extract data from all the raw systems/storages.
    All data is extracted to the intermediate layer, where names and types
    are adjusted.

    Returns:
        Kedro Pipeline object
    """
    pipe = Pipeline(
        [
            node(
                func=ingest_branches,
                inputs={},
                outputs="transactions.raw_branches@pandas",
                name="ingest_branches",
                tags=["ingestion"],
            ),
            node(
                func=ingest_promos,
                inputs={},
                outputs="transactions.raw_promos@pandas",
                name="ingest_promos",
                tags=["ingestion"],
            ),
            node(
                func=ingest_customers,
                inputs={},
                outputs="transactions.raw_customers@pandas",
                name="ingest_customers",
                tags=["ingestion"],
            ),
            node(
                func=ingest_vehicles,
                inputs={},
                outputs="transactions.raw_vehicles@pandas",
                name="ingest_vehicles",
                tags=["ingestion"],
            ),
            # node(
            #     func=ingest_invoices,
            #     inputs={},
            #     outputs="transactions.raw_invoices@pandas",
            #     name="ingest_invoices",
            #     tags=["ingestion"],
            # ),
            # node(
            #     func=ingest_invoices_items_PE,
            #     inputs={},
            #     outputs="transactions.raw_invoices_items_PE@pandas",
            #     name="ingest_invoices_items_PE",
            #     tags=["ingestion"],
            # ),
            # node(
            #     func=ingest_invoices_items_PAC,
            #     inputs={},
            #     outputs="transactions.raw_invoices_items_PAC@pandas",
            #     name="ingest_invoices_items_PAC",
            #     tags=["ingestion"],
            # ),
            # node(
            #     func=ingest_invoices_items,
            #     inputs={
            #         "invoicesitems_PE_df": "transactions.raw_invoices_items_PE@pandas",
            #         "invoicesitems_PAC_df": "transactions.raw_invoices_items_PAC@pandas",
            #     },
            #     outputs="transactions.raw_invoices_items@pandas",
            #     name="ingest_invoices_items",
            #     tags=["ingestion"],
            # ),
            # node(
            #     func=ingest_transactions,
            #     inputs={
            #         "invoice_df": "transactions.raw_invoices@pandas",
            #         "invoice_items_df": "transactions.raw_invoices_items@pandas",
            #     },
            #     outputs="transactions.raw_transactions@pandas",
            #     name="ingest_transactions",
            #     tags=["ingestion"],
            # ),
        ],
        tags=["ingestion"],
    )

    return pipe
