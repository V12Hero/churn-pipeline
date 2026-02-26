"""
This is a boilerplate pipeline 'predict_churn'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_with_conditions, predict_churn


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=filter_with_conditions,
                inputs={
                    # "df": "transactions.ftr_transactions@spark",
                    "df": "transactions.ftr_master@spark",
                    # "df": "transactions.segmentation_model_output@spark",
                    "conditions": "params:filter_churn_predict"
                },
                outputs="churn.mdl_predict_filtered@spark",
                name="filter_predict_churn",
                namespace="churn"
            ),
        node(
                func=predict_churn,
                inputs={
                    "df": "churn.mdl_predict_filtered@pandas",
                    "selected_cols": "params:features",
                    "model": "churn.mdl_estimator",
                    "threshold": "params:probability_threshold"
                },
                outputs="churn.mdl_churn_predicted",
                name="predict_churn",
                namespace="churn"
            ),
    ])
