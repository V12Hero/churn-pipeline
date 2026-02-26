"""
This is a boilerplate pipeline 'nptb_push'
generated using Kedro 0.18.12
"""
import pyspark
import pyspark.sql.functions as f
import pandas as pd
import scipy.sparse as sp
from dateutil.relativedelta import relativedelta
from implicit.als import AlternatingLeastSquares
from typing import Dict, List

import tqdm

def prepare_data_push(
        int_transactions:pyspark.sql.DataFrame,
        n_months_nptb_push: int,
        last_date,
        minimum_orders_to_be_considered: int,
        products_to_drop_push: List
):
    push_transactions = int_transactions.filter(
        ~f.col('product_id').isin(products_to_drop_push)
    )

    initial_date_n_months_before = pd.to_datetime(last_date) - relativedelta(months=n_months_nptb_push)

    push_transactions_grouped = push_transactions.filter(
        f.col('_observ_end_dt') >= initial_date_n_months_before
    ).groupBy(
        ['_id', 'product_code']
    ).agg(
        f.countDistinct('_observ_end_dt').alias('orders')
    ).filter(
        f.col('orders') > minimum_orders_to_be_considered
    )

    # breakpoint()

    return push_transactions_grouped

def create_sparse(
        transaction_grouped: pd.DataFrame,
):
    customer_index = {
        customer: index
        for index, customer in enumerate(transaction_grouped["_id"].unique())
    }

    item_index = {
        item: index for index, item in enumerate(transaction_grouped["product_code"].unique())
    }

    row_indices = [
        customer_index[customer] 
        for customer in transaction_grouped["_id"]
    ]
    col_indices = [
        item_index[item]
        for item in transaction_grouped["product_code"]
    ]

    index_customer = {
        index: customer
        for customer, index in customer_index.items()
    }
    index_item = {
        index: item 
        for item, index in item_index.items()
    }

    # breakpoint()

    sparse_matrix = sp.coo_matrix(
        (list(transaction_grouped.orders.values), (row_indices, col_indices))
    )
    train_matrix = sparse_matrix.tocsr()

    return row_indices, index_item, index_customer, train_matrix

def get_recommendations(
        row_indices,
        index_item,
        index_customer,
        train_matrix,
        als_params: Dict,
        recommendations_per_user: int,
):

    model = AlternatingLeastSquares(**als_params)
    model.fit(train_matrix)

    unique = set(row_indices)

    total = len(unique)

    rank_list = [i for i in range(1, recommendations_per_user + 1)]
    usuarios = []
    categorias = []
    ranking = []
    propensity = []

    for user in tqdm.tqdm(unique):
        ids, scores = model.recommend(
            user,
            train_matrix[user],
            N=recommendations_per_user,
            filter_already_liked_items=True,  ## Recomienda solo prductos nuevos
        )

        item_names = [index_item[index] for index in ids]
        usuarios += [index_customer[user]] * recommendations_per_user
        categorias += item_names
        ranking += rank_list
        propensity += [i for i in scores]

    recomendaciones_por_usuario = pd.DataFrame(
        {
            "_id": usuarios,
            "product_code_push": categorias,
            "ranking": ranking,
        }
    )

    return recomendaciones_por_usuario
