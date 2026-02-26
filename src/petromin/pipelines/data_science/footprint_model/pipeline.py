"""Model Segmentation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from petromin.modelling.preprocessing.filter_spark import pre_processing
from petromin.modelling.preprocessing.target import create_footprint_normalized_target
from petromin.reporting.html_report import create_html_report

from .nodes import (
    footprint_model_fit,
    footprint_model_hypertune,
    footprint_model_inference_test,
    footprint_model_inference_train,
    footprint_model_interpretability,
)


def create_pipeline(**kwargs) -> Pipeline:
    """It creates a pipeline for each model in the list of namespaces.

    Returns:
      Sum of Pipelines for all namespaces.
    """
    return sum([_create_pipeline(namespace) for namespace in kwargs.get("namespaces", [])])


def _create_pipeline(namespace: str) -> Pipeline:
    footprint_pipe = Pipeline(
        [
            # TODO: Ingrid Reis -- > Make sure that in the preprocessing you only use points from TecnoStores
            node(
                func=pre_processing,
                inputs={
                    "df": "ftr_master",
                    "conditions": "params:filter_footprint_model_conditions",
                },
                outputs="footprint_preprocessed@spark",
                name="footprint_preprocessed_master_table",
            ),
            node(
                func=create_footprint_normalized_target,
                inputs={
                    "df": "footprint_preprocessed@spark",
                    "params": "params:footprint.target_creation",
                },
                outputs="footprint_master_filtered@spark",
                name="footprint_master_table",
            ),
            node(
                func=footprint_model_hypertune,
                inputs={
                    "df": "footprint_master_filtered@spark",
                    "params": "params:footprint",
                },
                outputs={
                    "study": "footprint_model_study",
                    "best_trial_params": "footprint_best_trial_params",
                    # Test metrics comes from the hypertune cross validation exercise
                    "cross_validation_metrics": "footprint_model_test_metrics",
                },
                name="hypertune_footprint_model",
            ),
            node(
                func=footprint_model_fit,
                inputs={
                    "df": "footprint_master_filtered@spark",
                    "best_params": "footprint_best_trial_params",
                },
                outputs="footprint_model_artifact",
                name="fit_footprint_model",
            ),
            node(
                func=footprint_model_inference_train,
                inputs={
                    "df": "footprint_master_filtered@spark",
                    "model": "footprint_model_artifact",
                    "best_params": "footprint_best_trial_params",
                },
                outputs={
                    "df": "footprint_model_train_output",
                    "metrics": "footprint_model_train_metrics",
                },
                name="inference_footprint_model",
            ),
            node(
                func=footprint_model_interpretability,
                inputs={
                    "model": "footprint_model_artifact",
                },
                outputs={
                    "feature_importance": "footprint_feature_importance",
                },
                name="interpretability_footprint_model",
            ),
            node(
                func=footprint_model_inference_test,
                inputs={
                    "df": "prm_geolocation_footprint",
                    "model": "footprint_model_artifact",
                },
                outputs="footprint_model_test_inference",
                name="footprint_model_inference_test",
            ),
            node(
                func=create_html_report,
                inputs=[
                    "params:reporting.footprint.geographic_html_report",
                    "footprint_model_test_inference",
                ],
                outputs=[
                    "geographic_footprint_report",
                    "geographic_footprint_report_notebook_error",
                ],
                name="footprint_geographic_report",
            ),
        ]
    )

    inputs = {
        "ftr_master": "ftr_master",
        "prm_geolocation_footprint": "prm_geolocation_footprint",
    }
    return pipeline(
        pipe=footprint_pipe,
        namespace=namespace,
        inputs=inputs,
        outputs={
            "footprint_preprocessed@spark": f"{namespace}.footprint_preprocessed@spark",
            "footprint_master_filtered@spark": f"{namespace}.footprint_master_filtered@spark",
            "footprint_model_study": f"{namespace}.footprint_model_study",
            "footprint_best_trial_params": f"{namespace}.footprint_best_trial_params",
            "footprint_model_test_metrics": f"{namespace}.footprint_model_test_metrics",
            "footprint_model_artifact": f"{namespace}.footprint_model_artifact",
            "footprint_model_train_output": f"{namespace}.footprint_model_train_output",
            "footprint_model_train_metrics": f"{namespace}.footprint_model_train_metrics",
            "footprint_feature_importance": f"{namespace}.footprint_feature_importance",
            "footprint_model_test_inference": f"{namespace}.footprint_model_test_inference",
            "geographic_footprint_report": f"{namespace}.geographic_footprint_report",
        },
        parameters={},
    )
