"""Model Segmentation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_segmentation_excel_report


def create_pipeline(**kwargs) -> Pipeline:
    """It creates a pipeline for each model in the list of namespaces.

    Returns:
      Sum of Pipelines for all namespaces.
    """
    return sum([_create_pipeline(namespace) for namespace in kwargs.get("namespaces", [])])


def _create_pipeline(namespace: str) -> pipeline:
    reporting_pipe = Pipeline(
        [
            node(
                func=compute_segmentation_excel_report,
                inputs={
                    "df": "segmentation_model_output@pandas",
                    "params": "params:reporting.segmentation",
                    "model_params": "params:model_artifact",
                },
                outputs="segmentation_excel_report",
                name="segmentation_excel_report",
                tags=["segmentation_model", namespace],
            ),
        ]
    )
    inputs = {}
    return pipeline(
        pipe=reporting_pipe,
        namespace=namespace,
        inputs=inputs,
        outputs={
            "segmentation_excel_report": f"{namespace}.segmentation_excel_report",
        },
        parameters={},
    )
