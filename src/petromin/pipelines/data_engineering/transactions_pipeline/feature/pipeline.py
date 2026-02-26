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

from segmentation_core.helpers.data_processing.general import join_dfs
# from segmentation_core.helpers.parameters.load import load_parameters
from transactions.last_order_features import last_order_features
from transactions.nodes import (
    build_transactional_features_df,
    calculate_percentage_columns,
    create_business_features,
)
from transactions.utils import select_cols

# params = load_parameters()


def create_pipeline(namespace) -> Pipeline:
    """Create the pipeline for the transactions master table.

    kedro run --tag {namespace} 
    kedro run --tag "{namespace}_feature"
    kedro run --tag "{namespace}_join"

    e.g

    kedro run --tag "transactions_120d_feature"
    """
    dfs_list = [
        "per_customer",
        "per_customer_product_group",
        "per_customer_product_micro_category",
        "micro_category_activity"
    ] #params[namespace]["dfs"]

    nodes_list = []

    # Calculating feature for each different index datasets
    for df in dfs_list:
        nodes_list.append(
            node(
                func=build_transactional_features_df,
                inputs=[
                    "transactions_with_evaluation_date",
                    f"params:dfs.{df}.groupby_cols",
                    f"params:dfs.{df}.features",
                    f"params:dfs.{df}.num_days",
                    "params:global_params",
                ],
                outputs=f"feature.{df}",
                name=f"build_transactional_features_df.{df}",
                tags=[
                    "transactions",
                    f"{namespace}_features",
                    f"{namespace}_{df}_features",
                ],
            )
        )

    # Create last order features
    nodes_list += [
        node(
            func=last_order_features,
            inputs=[
                "transactions_with_evaluation_date",
                "params:per_customer_per_sku_last_order",
            ],
            outputs="feature.per_customer_per_sku_last_order",
            name="per_customer_per_sku_last_order",
            tags=[
                "transactions",
                f"{namespace}_features",
            ],
        ),
        node(
            func=last_order_features,
            inputs=[
                "transactions_with_evaluation_date",
                "params:per_customer_last_order",
            ],
            outputs="feature.per_customer_last_order",
            name="per_customer_last_order",
            tags=[
                "transactions",
                f"{namespace}_features",
            ],
        ),
        node(
            func=select_cols,
            inputs=[
                "feature.per_customer_last_order",
                "params:select_cols.cols_to_select",
                "params:select_cols.cols_to_drop",
            ],
            outputs="feature.per_customer_last_order_selected",
            name="per_customer_last_order_selected",
            tags=[
                "transactions",
                f"{namespace}_features",
            ],
        ),
    ]

    # Joining DFs with the calculated features
    #
    nodes_list += [
        node(
            join_dfs,
            inputs=["params:join_col"]
            + [f"feature.{df}" for df in dfs_list]
            + ["feature.per_customer_last_order_selected"],
            outputs="feature.join_dfs_features",
            tags=["transactions", f"{namespace}_join"],
            name="join_dfs",
        ),
        node(
            calculate_percentage_columns,
            inputs=["feature.join_dfs_features", "params:calculate_percentages"],
            outputs="feature.transactions_percentage",
            tags=["transactions", f"{namespace}_join"],
            name="calculate_percentage_columns",
        ),
        node(
            create_business_features,
            inputs=[
                "feature.transactions_percentage",
                "params:business_features",
            ],
            outputs="feature",
            name="create_business_features",
            tags=["transactions", f"{namespace}_join"],
        ),
    ]

    return pipeline(
        nodes_list,
        inputs={
            "transactions_with_evaluation_date": "transactions.transactions_with_evaluation_date"
        },
        tags=[
            "transactions",
            "transactions_de",
            "feature",
            "transactions_feature",
            namespace,
        ],
        namespace=namespace,
    )
