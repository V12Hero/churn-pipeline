"""
This is a boilerplate pipeline 'model_churn'
generated using Kedro 0.18.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import filter_with_conditions, \
                    drop_columns, \
                    scale_data, \
                    get_panel_data_cross_validation_indexes, \
                    tune_hyperparameters, \
                    select_features, \
                    tune_hyperparameters, \
                    drop_columns, \
                    train_model, \
                    validation_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_with_conditions,
                inputs={
                    "df": "transactions.ftr_master@spark",
                    # "df": "transactions.ftr_transactions@spark",
                    # "df": "transactions.ftr_churn@spark",
                    # "df": "transactions.segmentation_model_output@spark",
                    "conditions": "params:filter_model_conditions"
                },
                outputs="churn.mdl_master_filtered@spark",
                name="filter_master_table",
            ),
            node(
                func=drop_columns,
                inputs={
                    "df": "churn.mdl_master_filtered@spark",
                    "cols_to_drop": "params:cols_to_drop",
                    "special_cols": "params:special_cols",
                },
                outputs="churn.mdl_master_without_unused_cols@spark",
                name="drop_unused_columns",
                namespace="churn"
            ),
            # node(
            #     func=scale_data,
            #     inputs={
            #         "df": "mdl_master_without_unused_cols@pandas",
            #         "target": "params:target",
            #         "special_cols": "params:special_cols",
            #     },
            #     outputs=["scaler","mdl_master_scaled"],
            #     name="scale_data",
            # ),
            node(
                func=get_panel_data_cross_validation_indexes,
                inputs={
                    "data": "churn.mdl_master_without_unused_cols@pandas",
                    "splits": "params:cv_splits",
                    "date_column": "params:cv_date_column",
                    "months_gap": "params:cv_months_gap",
                    "max_train_size": "params:max_train_size",
                    "months_test": "params:cv_months_test",
                    "undersampling_rate": "params:undersampling_rate",
                    "target_column": "params:target"
                },
                outputs="churn.mdl_cv_indexes",
                name="generate_cv_indexes",
                namespace="churn"
            ),
            node(
                func=select_features,
                inputs={
                  "data": "churn.mdl_master_without_unused_cols@pandas",
                  "universe_of_features": "params:universe_of_features",
                  "target": "params:target",
                  "cv": "churn.mdl_cv_indexes",
                  "n_features_to_select": "params:n_features_to_select",
                  "direction": "params:direction",
                  "scoring": "params:tuning_metric_to_maximize"
                },
                outputs="churn.mdl_selected_features",
                name="select_features",
                namespace="churn"
            ),
            # node(
            #     func=tune_hyperparameters,
            #     inputs={
            #         "data": "mdl_master_without_unused_cols@pandas",
            #         "parameters_distribution": "params:tuning_params_distribution",
            #         "cv": "mdl_cv_indexes",
            #         "features": "params:features", # To modify with selected features after the feature selection
            #         "target": "params:target",
            #         "iterations": "params:tuning_iterations",
            #         "scoring": "params:tuning_scoring",
            #         "metric_to_maximize": "params:tuning_metric_to_maximize"
            #     },
            #     outputs={"tuning_results": "mdl_tuning_results",
            #              "tuning_parameters": "mdl_tuning_params",
            #              "tuning_estimator": "mdl_tuning_estimator",
            #     },
            #     name="tune_hyperparameters",
            # ),
            node(
                func=train_model,
                inputs={
                    "data": "churn.mdl_master_without_unused_cols@pandas",
                    # "columns": "params:universe_of_features",
                    # "columns": "params:features",
                    "columns": "params:selected_features",
                    "model_hyperparameters":  "params:universe_of_features", #"mdl_tuning_params",
                    "threshold": "params:probability_threshold", # To modify with selected features after the feature selection
                    "target_column": "params:target",
                    "cv_indexes": "churn.mdl_cv_indexes"
                }, 
                outputs=[
                    "churn.mdl_churn_avg_metrics", 
                    "churn.mdl_churn_metrics_df",
                    "churn.mdl_estimator",
                    "churn.mdl_explainer"
                ],
                name="train_churn_model",
                namespace="churn"
            ),node(
                func=filter_with_conditions,
                inputs={
                    # "df": "transactions.ftr_transactions@spark",
                    # "df": "transactions.ftr_churn@spark",
                    "df": "transactions.segmentation_model_output@spark",
                    "conditions": "params:filter_churn_validation"
                },
                outputs="churn.mdl_validation_filtered@spark",
                name="filter_validation_churn",
                namespace="churn"
            ),
            node(
                func=validation_node,
                inputs={
                    "data": "churn.mdl_validation_filtered@pandas",
                    # "columns": "params:universe_of_features",
                    # "columns": "params:features",
                    "columns": "params:selected_features",
                    "model": "churn.mdl_estimator",
                    "explainer": "churn.mdl_explainer",
                    "target_column": "params:target",
                    "threshold": "params:probability_threshold", # To modify with selected features after the feature selection
                }, 
                outputs=[
                    "churn.mdl_churn_validation_avg_metrics",
                    "churn.mdl_churn_validation_metrics_df",
                    "churn.mdl_churn_validation_beeswarm_plot",
                    "churn.mdl_churn_validation_average_importance_plot",
                    "churn.mdl_churn_validation_feature_importance",
                ],
                name="validation_node",
                namespace="churn"
            ),
        ]
    )
