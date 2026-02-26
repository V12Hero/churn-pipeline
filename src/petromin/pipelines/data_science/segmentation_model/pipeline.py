"""Model Segmentation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from petromin.modelling.preprocessing.filter_spark import pre_processing
from petromin.reporting.html_report import create_html_report

from .nodes import segmentation_model_fit, segmentation_model_inference, segmentation_model_inference_spark, ftr_join_dfs_segmentations


def create_pipeline(inference=False, **kwargs) -> Pipeline:
    """It creates a pipeline for each model in the list of namespaces.

    Returns:
      Sum of Pipelines for all namespaces.
    """
    if inference == False:
        namespaces_pipeline = sum([_create_pipeline(namespace) for namespace in kwargs.get("namespaces", [])])

        out = namespaces_pipeline
    else:
        namespaces_pipeline = sum([_create_pipeline_inference(namespace) for namespace in kwargs.get("namespaces", [])])

        joined_namespaces_inputs = {f"{namespace}": f"{namespace}.segmentation_model_output@spark" for namespace in kwargs.get("namespaces", [])}

        master_pipeline = Pipeline([
            node(
                    func=ftr_join_dfs_segmentations,
                    inputs=joined_namespaces_inputs,
                    outputs="transactions.segmentation_model_output@spark",
                    name="join_all_segmentations",
                    tags=["prediction", "all"]
                ),
            ],
            tags=["segmentation_model"],
        )
        out = namespaces_pipeline + master_pipeline


    return out


def _create_pipeline(namespace: str) -> pipeline:
    segmentation_pipe = Pipeline(
        [
            node(
                func=pre_processing,
                inputs={
                    "df": "ftr_master@spark",
                    "conditions": "params:filter_segmentation_model_conditions",
                },
                outputs="segmentation_master_filtered@spark",
                name="segmentation_filter_master_table",
            ),
            node(
                func=segmentation_model_fit,
                inputs={
                    "df": "segmentation_master_filtered@pandas",
                    "params": "params:model_artifact",
                },
                outputs={
                    "pipeline": "segmentation_model_artifact",
                    "params": "segmentation_model_params_artifact",
                    "fig": "inertia_cluster_plot",
                    "centroids": "segmentation_centroids",
                    "wcss_feature_importance": "segmentation_wcss_feature_importance",
                    "unsupervised_feature_importance": "segmentation_unsupervised_feature_importance",
                },
                name="segmentation_model_fit",
            ),
            # node(
            #     func=pre_processing,
            #     inputs={
            #         "df": "ftr_master@spark",
            #         "conditions": "params:filter_segmentation_prediction_conditions",
            #     },
            #     outputs="segmentation_master_filtered_prediction@spark",
            #     name="segmentation_filter_master_table_prediction",
            #     tags=["prediction", "all"]
            # ),
            # node(
            #     func=segmentation_model_inference,
            #     inputs={
            #         "df": "segmentation_master_filtered_prediction@pandas",
            #         # "df": "ftr_master@pandas",
            #         "params": "params:model_artifact",
            #         "pipeline": "segmentation_model_artifact",
            #     },
            #     outputs="segmentation_model_output@pandas",
            #     name="segmentation_model_inference",
            #     tags=["prediction", "all"]
            # ),
            # node(
            #     func=segmentation_model_inference_spark,
            #     inputs={
            #         "df": "segmentation_master_filtered@spark",
            #         # "df": "ftr_master@pandas",
            #         "params": "params:model_artifact",
            #         "pipeline": "segmentation_model_artifact",
            #     },
            #     outputs="segmentation_model_output@spark",
            #     name="segmentation_model_inference",
            # ),
            # node(
            #     func=create_html_report,
            #     inputs=[
            #         "params:reporting.segmentation.geographic_html_report",
            #         "segmentation_model_output@pandas",
            #     ],
            #     outputs=[
            #         "geographic_cluster_report",
            #         "geographic_cluster_report_notebook_error",
            #     ],
            #     name="segmentation_geographic_report",
            #     tags=["segmentation_geographic_report"],
            # ),
            # node(
            #     func=create_html_report,
            #     inputs=[
            #         "params:reporting.segmentation.transactional_html_report",
            #         "geographic_cluster_report",
            #     ],
            #     outputs=[
            #         "transactional_cluster_report",
            #         "transactional_cluster_report_notebook_error",
            #     ],
            #     name="segmentation_transactional_report",
            # ),
        ],
        tags=["segmentation_model", namespace],
    )
    # Inputs datasets
    inputs = {
        "ftr_master": "transactions.ftr_master",
        # "ftr_master": "transactions.ftr_transactions",
        # "ftr_master": "transactions.ftr_churn",
    }
    return pipeline(
        pipe=segmentation_pipe,
        namespace=namespace,
        inputs=inputs,
        outputs={
            "segmentation_master_filtered@spark": f"{namespace}.segmentation_master_filtered@spark",
            "segmentation_model_artifact": f"{namespace}.segmentation_model_artifact",
            "segmentation_model_params_artifact": f"{namespace}.segmentation_model_params_artifact",
            # "segmentation_model_output@pandas": f"{namespace}.segmentation_model_output@pandas",
            # "geographic_cluster_report": f"{namespace}.geographic_cluster_report",
            # "transactional_cluster_report": f"{namespace}.transactional_cluster_report",
            "inertia_cluster_plot": f"{namespace}.inertia_cluster_plot",
            "segmentation_centroids": f"{namespace}.segmentation_centroids",
        },
        parameters={},
    )


def _create_pipeline_inference(namespace: str) -> pipeline:
    segmentation_pipe = Pipeline(
        [
            # node(
            #     func=pre_processing,
            #     inputs={
            #         "df": "ftr_master@spark",
            #         "conditions": "params:filter_segmentation_model_conditions",
            #     },
            #     outputs="segmentation_master_filtered@spark",
            #     name="segmentation_filter_master_table",
            # ),
            # node(
            #     func=segmentation_model_fit,
            #     inputs={
            #         "df": "segmentation_master_filtered@pandas",
            #         "params": "params:model_artifact",
            #     },
            #     outputs={
            #         "pipeline": "segmentation_model_artifact",
            #         "params": "segmentation_model_params_artifact",
            #         "fig": "inertia_cluster_plot",
            #         "centroids": "segmentation_centroids",
            #         "wcss_feature_importance": "segmentation_wcss_feature_importance",
            #         "unsupervised_feature_importance": "segmentation_unsupervised_feature_importance",
            #     },
            #     name="segmentation_model_fit",
            # ),
            node(
                func=pre_processing,
                inputs={
                    "df": "ftr_master@spark",
                    "conditions": "params:filter_segmentation_prediction_conditions",
                },
                outputs="segmentation_master_filtered_prediction@spark",
                name="segmentation_filter_master_table_prediction",
                tags=["prediction", "all"]
            ),
            node(
                func=segmentation_model_inference,
                inputs={
                    "df": "segmentation_master_filtered_prediction@pandas",
                    # "df": "ftr_master@pandas",
                    "params": "params:model_artifact",
                    "pipeline": "segmentation_model_artifact",
                },
                outputs="segmentation_model_output@pandas",
                name="segmentation_model_inference",
                tags=["prediction", "all"]
            ),
            # node(
            #     func=segmentation_model_inference_spark,
            #     inputs={
            #         "df": "segmentation_master_filtered@spark",
            #         # "df": "ftr_master@pandas",
            #         "params": "params:model_artifact",
            #         "pipeline": "segmentation_model_artifact",
            #     },
            #     outputs="segmentation_model_output@spark",
            #     name="segmentation_model_inference",
            # ),
            # node(
            #     func=create_html_report,
            #     inputs=[
            #         "params:reporting.segmentation.geographic_html_report",
            #         "segmentation_model_output@pandas",
            #     ],
            #     outputs=[
            #         "geographic_cluster_report",
            #         "geographic_cluster_report_notebook_error",
            #     ],
            #     name="segmentation_geographic_report",
            #     tags=["segmentation_geographic_report"],
            # ),
            # node(
            #     func=create_html_report,
            #     inputs=[
            #         "params:reporting.segmentation.transactional_html_report",
            #         "geographic_cluster_report",
            #     ],
            #     outputs=[
            #         "transactional_cluster_report",
            #         "transactional_cluster_report_notebook_error",
            #     ],
            #     name="segmentation_transactional_report",
            # ),
        ],
        tags=["segmentation_model", namespace],
    )
    # Inputs datasets
    inputs = {
        "ftr_master": "transactions.ftr_master",
        # "ftr_master": "transactions.ftr_transactions",
        # "ftr_master": "transactions.ftr_churn",
    }
    return pipeline(
        pipe=segmentation_pipe,
        namespace=namespace,
        inputs=inputs,
        outputs={
            # "segmentation_master_filtered@spark": f"{namespace}.segmentation_master_filtered@spark",
            # "segmentation_model_artifact": f"{namespace}.segmentation_model_artifact",
            # "segmentation_model_params_artifact": f"{namespace}.segmentation_model_params_artifact",
            "segmentation_model_output@pandas": f"{namespace}.segmentation_model_output@pandas",
            # "geographic_cluster_report": f"{namespace}.geographic_cluster_report",
            # "transactional_cluster_report": f"{namespace}.transactional_cluster_report",
            # "inertia_cluster_plot": f"{namespace}.inertia_cluster_plot",
            # "segmentation_centroids": f"{namespace}.segmentation_centroids",
        },
        parameters={},
    )
