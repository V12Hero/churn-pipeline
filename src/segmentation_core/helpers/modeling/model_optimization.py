# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.

import logging
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from yellowbrick.cluster import KElbowVisualizer

from .modeling.evaluation_metrics import (
    _build_metrics_ranking,
    _get_clustering_performance_metrics,
)
from .objects.load import load_object
from .reproductibility.seed_file import get_global_seed, seed_file

logger = logging.getLogger(__name__)

sns.set_style("darkgrid")
sns.set_palette("crest_r")
sns.set(rc={"figure.dpi": 100, "savefig.dpi": 100, "figure.figsize": (12, 8)})


def cluster_inference(
    df: pd.DataFrame, features: List, model: Pipeline, cluster_col: str
) -> pd.DataFrame:
    """Do cluster inference.

    Perform cluster inference, first try .predict method and then .fit_predict.
    (Clustering algorithms have differents inferences methods)

    Args:
        df (pd.DataFrame): Data to perform cluster inference
        features (List): List of features
        model (sklearn.cluster): Cluster model
        cluster_col (str): Name of the cluster column.

    Returns:
        pd.DataFrame:
    """
    try:
        df[cluster_col] = model.predict(df[features])
    except Exception:
        df[cluster_col] = model.fit_predict(df[features])
    return df


def train_cluster_model(df: pd.DataFrame, model_params: Dict) -> pd.DataFrame:
    """Train cluster algorithms.

    Train cluster algorithms and return df with the cluster columnn.

    Model params should follow the above structure below:

        {'class': 'sklearn.cluster.CLUSTER_NAME',
        'cluster_col': 'cluster',
        'features': ['feature1',
                    'feature2',
                    'feature3',
                    'feature4',
                    .....],
        'id_col': 'customer_id',
        'kwargs': {'n_clusters': N, 'random_state': GLOBAL_SEED},
        'scaler_params': {'class': 'sklearn.preprocessing.SCALER_NAME',
                        'kwargs': None}}

    Args:
        df (pd.DataFrame): Master table encoded.
        model_params (Dict): Model params.

    Returns:
        Dict: Model pipeline and df with cluster col.
    """
    cluster_model = load_object(model_params)
    scaler_object = load_object(model_params["scaler_params"])
    model_pipeline = Pipeline(
        [("scaler", scaler_object), ("cluster_model", cluster_model)]
    )
    features = model_params["features"]
    cluster_col = model_params["cluster_col"]
    logger.info(f"Model pipeline: {model_pipeline}")
    df_with_cluster = cluster_inference(df, features, model_pipeline, cluster_col)
    output_dict = {"trained_model": model_pipeline, "df_with_cluster": df_with_cluster}
    return output_dict


def train_cluster_with_elbow_method(
    df: pd.DataFrame,
    model_sel_params: Dict,
    model_params: Dict = None,
    selected_cols: list = [],
) -> pd.DataFrame:
    """Clustering optimization.

    1. Find optimal number of clusters.
    2. Fitt final model with this number.

    It takes a dataframe, a dictionary of parameters, and a list of columns to use as features, and
    returns a dataframe with a new column containing the cluster labels

    Args:
      df (pd.DataFrame): the dataframe to be clustered
      model_sel_params (Dict): This is the dictionary of parameters that we've been using throughout this
    notebook.
      selected_cols (list): list of columns to use for training the model
      model_params (Dict): Model params to instantiate clustering model.

    Returns:
      The dataframe with the cluster column added.
    """
    input_df = df.copy()
    # seed an params
    seed = get_global_seed()
    seed_file(seed, verbose=False)
    feature_blocklist = model_sel_params["feature_blocklist"]
    # Selecting features used to train the model
    columns = [col for col in selected_cols if col not in feature_blocklist]
    if len(columns) == 0:
        columns = [col for col in df.columns if col not in feature_blocklist]
    min_clusters = model_sel_params["clustering"]["min_clusters"]
    max_clusters = model_sel_params["clustering"]["max_clusters"]

    # load object, fit and scale data.
    scaler = load_object(model_sel_params["clustering"]["default_scaler"])
    scaler = scaler.fit(input_df[columns].values)
    input_df[columns] = scaler.transform(input_df[columns].values)

    # load model object
    if model_params is not None:
        cluster_model = load_object(model_params)
    else:
        cluster_model = load_object(model_sel_params["clustering"]["default_model"])
        model_params = model_sel_params["clustering"]["default_model"]

    # automatic k search or not
    if model_sel_params["clustering"]["automatically_search_number_of_clusters"]:
        visualizer = KElbowVisualizer(
            cluster_model,
            k=(min_clusters, max_clusters),
            timings=False,
            # metric="calinski_harabasz"
        )
        visualizer.fit(input_df[columns])  # Fit data to visualizer
        num_clusters = visualizer.elbow_value_
        logger.info(f"Using {num_clusters} clusters")
    else:
        num_clusters = model_sel_params["clustering"]["num_clusters"]
        logger.info(f"Using a fixed number of clusters: {num_clusters}")

    # Number of cluster check
    num_clusters = _check_valid_number_of_clusters(num_clusters, model_sel_params)

    # adjust model params and fit best model with the best number of clusters
    model_params["kwargs"]["n_clusters"] = num_clusters
    final_model = load_object(model_params)
    final_model.fit(input_df[columns])
    df[model_sel_params["cluster_col"]] = final_model.labels_

    final_model_pipe = Pipeline([("scaler", scaler), ("clustering_model", final_model)])

    return df, final_model_pipe


def _check_valid_number_of_clusters(num_clusters: Any, model_sel_params: Dict) -> int:
    """Returns a valid number of clusters.

    It may be the case that the optimum K is not between the min and max values set by
    k_min, k_max, if this is the case, the ElbowVisualizer will deliver a None result,
    which is not a suitable number for clustering. This function changes this None value
    to the maximum cluster value given in the model parameters.

    Args:
        num_clusters (Any): ElbowVisualizer num of cluster output.
        model_sel_params (Dict): Model selection params.

    Returns:
        int: _description_
    """
    if num_clusters is None:
        msg1 = "The elbow algorithm ended up not finding the optimal number of clusters K, "
        msg2 = "therefore the optimal number of clusters was set as the largest number of clusters."
        logger.warning(msg1 + msg2)
        num_clusters = model_sel_params["clustering"]["max_clusters"]
    return num_clusters


def _clustering_optimization(
    df: pd.DataFrame,
    model_sel_params: Dict,
    fs_params: Dict,
    total_selected_features: List,
) -> pd.DataFrame:
    """Clustering optimization.

    > This function runs the clustering optimization process for each model in the
    `model_wrapper_params` dictionary

    Args:
      df (pd.DataFrame): the dataframe to be used for clustering
      model_sel_params (Dict): The model selection parameters.
      fs_params (Dict): This is the parameters dictionary for the feature selection process.
      total_selected_features (List): List of features selected by the feature selection algorithm

    Returns:
      A dataframe with the best model for each feature selection method.
    """
    # wrapper models
    model_wrapper_params = model_sel_params["clustering"]["model_wrapper"]
    # metrics ponderation
    metrics_computation = fs_params["clustering_iteration_trees_importance_based"][
        "metrics_computation"
    ]
    # find best model based on Elbow Metric
    model_selection_report = pd.DataFrame()
    for model in model_wrapper_params["models"].keys():
        logger.info(f"Running clustering optimization for model: {model}")
        model_params = model_wrapper_params["models"][model]
        # metrics report
        report_df = _clustering_optimization_report(
            df, model_sel_params, fs_params, model_params, total_selected_features
        )
        report_df.insert(0, "model", "")
        report_df["model"] = model
        model_selection_report = pd.concat([model_selection_report, report_df], axis=0)
    model_selection_report = _build_metrics_ranking(
        model_selection_report, metrics_computation
    )
    return model_selection_report


def _clustering_optimization_report(
    df: pd.DataFrame,
    model_sel_params: Dict,
    fs_params: Dict,
    model_params: Dict,
    total_select_features: List,
) -> pd.DataFrame:
    """Clustering Optimization Report.

    This function takes in a dataframe, model selection parameters, feature selection parameters, and a
    list of selected features. It then iterates through the list of selected features, and for each
    iteration, it trains a clustering model using a "test selected" selected features,

    Its reports the optimal number of clusters, the number of features, the features, and the performance metrics of the
    clustering model.

    Args:
      df (pd.DataFrame): the dataframe that we want to cluster
      model_sel_params (Dict): the parameters for the clustering model
      fs_params (Dict): the parameters for feature selection
      total_select_features (List): list of features selected by the feature selection algorithm

    Returns:
      A dataframe with the optimal number of clusters, the number of features, the features, and the
    performance metrics.
    """
    min_num_features = fs_params["clustering_iteration_trees_importance_based"][
        "min_num_features"
    ]
    cluster_col = model_sel_params["cluster_col"]
    reports = []
    # nb of features selected
    for i in range(min_num_features, len(total_select_features) + 1):
        report = {}
        test_features = total_select_features[:i]
        input_df = df[test_features]
        # train clustering models
        df_with_clusters, _ = train_cluster_with_elbow_method(
            input_df,
            model_sel_params,
            model_params=model_params,
            selected_cols=test_features,
        )
        # reporting
        report["optimal_number_of_clusters"] = df_with_clusters[cluster_col].nunique()
        report["num_features"] = i
        report["features"] = test_features
        report.update(
            _get_clustering_performance_metrics(
                df_with_clusters, cluster_col, model_sel_params["feature_blocklist"]
            )
        )
        reports.append(report)
    report_df = pd.DataFrame(reports)
    return report_df


def train_clustering_wrapper(
    df: pd.DataFrame,
    model_sel_params: Dict,
    metrics_computation: Dict,
    selected_features: List,
) -> Dict:
    """Train clustering wrapper.

    It takes a dataframe, a dictionary of parameters, a dictionary of metrics, and a list of features,
    and returns a dictionary of a dataframe and a dataframe

    Args:
      df (pd.DataFrame): the dataframe to be clustered
      model_sel_params (Dict): This is the model selection parameters dictionary.
      metrics_computation (Dict): This is a dictionary that contains the metrics that you want to use to
    rank the models.
      selected_features (List): list of features to be used for clustering

    Returns:
      The output of the function is a dictionary with two keys:
        - df_cluster: the dataframe with the cluster column
        - model_ranking: the dataframe with the ranking of the models
    """
    model_wrapper_params = model_sel_params["clustering"]["model_wrapper"]
    cluster_col = model_sel_params["cluster_col"]
    # find best model based on Elbow Metric
    results = []
    cluster_dict = {}
    for model in model_wrapper_params["models"].keys():
        metrics = {}
        model_params = model_wrapper_params["models"][model]
        # initial clustering model
        logger.info(f"Optimizing {model} model ...")
        df_model_cluster, _ = train_cluster_with_elbow_method(
            df,
            model_sel_params,
            selected_cols=selected_features,
            model_params=model_params,
        )
        cluster_dict[model] = df_model_cluster
        metrics["model"] = model
        metrics["optimal_number_of_clusters"] = df_model_cluster[cluster_col].nunique()

        metrics.update(
            _get_clustering_performance_metrics(
                df_model_cluster, cluster_col, model_sel_params["feature_blocklist"]
            )
        )
        results.append(metrics)

    results = pd.DataFrame(results)
    df_rank = _build_metrics_ranking(results, metrics_computation)
    best_model = df_rank["model"].iloc[0]
    logger.info(f"Best model selected from the wrapper {best_model}")
    # data with clusters
    df_clusterized = cluster_dict[best_model]
    # final output dict
    output_dict = {"df_cluster": df_clusterized, "model_ranking": df_rank}
    return output_dict
