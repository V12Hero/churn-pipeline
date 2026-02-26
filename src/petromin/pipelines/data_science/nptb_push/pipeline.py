# """
# This is a boilerplate pipeline 'nptb_push'
# generated using Kedro 0.18.12
# """

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import prepare_data_push, create_sparse, get_recommendations


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_data_push,
                inputs={
                    'int_transactions': 'nptb_transaction_data',
                    'n_months_nptb_push': 'params:n_months_nptb_push',
                    'last_date': 'params:last_date_push',
                    'minimum_orders_to_be_considered': 'params:minimum_orders_to_be_considered_push',
                    'products_to_drop_push': 'params:products_to_drop_push'
                },
                outputs='nptb_push_transactions_grouped@spark',
                name="prepare_data_push"
            ),
            node(
                func=create_sparse,
                inputs={
                    'transaction_grouped': 'nptb_push_transactions_grouped@pandas'
                },
                outputs=['nptb_push_row_indices', 
                         'nptb_push_index_item',
                         'nptb_push_index_customer',
                         'nptb_push_train_matrix'],
                name="create_sparse_push"
            ),
            node(
                func=get_recommendations,
                inputs={
                    'row_indices': 'nptb_push_row_indices',
                    'index_item': 'nptb_push_index_item',
                    'index_customer': 'nptb_push_index_customer',
                    'train_matrix': 'nptb_push_train_matrix',
                    'als_params': 'params:nptb_push_als_params',
                    'recommendations_per_user': 'params:nptb_push_recommendations_per_user'
                },
                outputs='nptb_push_recommendations',
                name="nptb_push_get_recommendations"
            ),
        ]
        )
