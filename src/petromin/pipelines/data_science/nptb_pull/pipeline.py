"""
This is a boilerplate pipeline 'nptb_pull'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import prepare_transactions, \
                   nptb_pull_create_features, \
                   nptb_clean_transactions_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nptb_clean_transactions_data,
                inputs={'prm_spine': 'transactions.prm_spine',
                        'int_transactions': 'transactions.int_transactions',
                        'categories_to_drop': 'params:categories_to_drop',
                        'products_to_drop': 'params:products_to_drop',
                        },
                outputs='nptb_transaction_data',
                name="nptb_clean_transactions_data"
            ),
            node(
                func=prepare_transactions,
                inputs={'base_transactions': 'nptb_transaction_data',
                        'n_months_nptb_pull': 'params:n_months_nptb_pull',
                        'initial_date': 'params:initial_date_nptb_pull'
                        },
                outputs='nptb_pull_transaction',
                name="nptb_pull_prepare_trx_data"
            ),
            node(
                func=nptb_pull_create_features,
                inputs={
                    'prm_spine': 'transactions.prm_spine',
                    'transactions': 'nptb_pull_transaction',
                    'n_months_nptb_pull': 'params:n_months_nptb_pull',
                    'n_recommendations_pull': 'params:n_recommendations_pull',
                    'min_number_of_trx_for_recommendations': 'params:min_number_of_trx_for_recommendations_pull'},
                outputs='nptb_pull_transaction_features',
                name="nptb_pull_create_features"
            ),
        ])
