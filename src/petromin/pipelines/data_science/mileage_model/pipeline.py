"""
This is a boilerplate pipeline 'model_mileage'
generated using Kedro 0.18.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_with_conditions, prepare_mileage_forecast, forecast_mileage


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_with_conditions,
                inputs={
                    "df": "transactions.ftr_master@spark",
                    "conditions": "params:filter_mileage_forecast"
                },
                outputs="mileage.forecast_master_filtered",
                name="filter_mileage_forecast",
                namespace="mileage"
            ),
            node(
                func=prepare_mileage_forecast,
                inputs={
                    "ftr_master": "mileage.forecast_master_filtered",
                },
                outputs="mileage.forecast_base@spark",
                name="prepare_mileage_forecast",
                namespace="mileage"
            ),
            node(
                func=forecast_mileage,
                inputs={
                    "forecast_df": "mileage.forecast_base@pandas",
                    "churn_df": "churn.mdl_churn_predicted",
                    "closed_station_list": "params:mileage.closed_station_list",
                }, 
                outputs=[
                    "mileage.mileage_forecast",
                ],
                name="forecast_mileage",
                namespace="mileage"
            ),
        ]
    )
