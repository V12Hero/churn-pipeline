"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import petromin.pipelines.data_engineering.data_engineering_pipeline as de_pipeline
import petromin.pipelines.data_engineering.users_pipeline as users_pipe
import petromin.pipelines.data_engineering.mastertable_pipeline as mastertable_pipe

import petromin.pipelines.data_engineering.general_pipeline.raw.general as tag_dict_pipe
import petromin.pipelines.data_engineering.transactions_pipeline as transactions_pipe
import petromin.pipelines.data_engineering.worldpop_pipeline as worldpop_pipe

import petromin.pipelines.data_engineering.new_transactions_pipeline as transactions_pipe
import petromin.pipelines.data_engineering.ingestion_pipeline as ingestion_pipe

from petromin.pipelines.data_science import (
    footprint_model,
    reporting,
    segmentation_model,
    churn_model,
    mileage_model,
    predict_churn,
    nptb_pull,
    nptb_push
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())

    # namespaces definition
    segmentation_namespaces = ["loyal", "pot_loyal", "uncommited", "new_joiner"]
    churn_namespaces = ["churn"]
    footprint_namespaces = ["petromin"]

    # data science
    segmentation_training_pipe = segmentation_model.create_pipeline(
        inference=False,
        namespaces=segmentation_namespaces
    )

    segmentation_inference_pipe = segmentation_model.create_pipeline(
        inference=True,
        namespaces=segmentation_namespaces
    )

    churn_pipe = churn_model.create_pipeline(
        # namespaces=churn_namespaces
    )

    predict_churn_pipe = predict_churn.create_pipeline(
        # namespaces=churn_namespaces
    )

    mileage_pipe = mileage_model.create_pipeline(
        # namespaces=churn_namespaces
    )

    nptb_pull_pipe = nptb_pull.create_pipeline()
    nptb_push_pipe = nptb_push.create_pipeline()

    nptb_pipe = nptb_pull_pipe + nptb_push_pipe

    # + reporting.create_pipeline(namespaces=segmentation_namespaces)

    # footprint_model_pipe = footprint_model.create_pipeline(namespaces=footprint_namespaces)

    # reporting
    # reporting_pipe = reporting.create_pipeline(namespaces=segmentation_namespaces)

    # tag_dict = tag_dict_pipe.create_pipeline()
    users_pipeline = users_pipe.create_pipeline()
    # TODO REVIEW PIPELINE
    de_pipe = de_pipeline.create_pipeline()
    transactions_pipeline = transactions_pipe.create_pipeline()
    ingestion_pipeline = ingestion_pipe.create_pipeline()
    worldpop_pipeline = worldpop_pipe.create_pipeline()

    mastertable_pipeline = mastertable_pipe.create_pipeline()

    pipelines_dict = {
        # "__default__": de_pipe,
        "de_pipeline": de_pipe,
        "users_pipeline": users_pipeline,
        "osm_pipeline": de_pipe.only_nodes_with_tags("osm_de", "tag_dict"),
        "tag_pipeline": de_pipe.only_nodes_with_tags("general"),
        # TODO: review this pipeline
        "spatial_pipeline": de_pipe.only_nodes_with_tags("spatial_mastertable"),
        "ingestion_pipeline": ingestion_pipeline,
        "transactions_pipeline": transactions_pipeline,
        "worldpop_pipeline": worldpop_pipeline,
        "mastertable_pipeline": mastertable_pipeline,

        # Data Science
        "segmentation_model": segmentation_training_pipe,
        "segmentation_inference": segmentation_inference_pipe,
        "churn_model": churn_pipe,
        "predict_churn": predict_churn_pipe,
        "mileage_model": mileage_pipe,
        "nptb_pull": nptb_pull_pipe,
        "nptb_push": nptb_push_pipe,
        "nptb_pipeline": nptb_pipe,
        "full_mileage_model": transactions_pipeline + predict_churn_pipe + mileage_pipe
    }

    return pipelines_dict #pipelines
