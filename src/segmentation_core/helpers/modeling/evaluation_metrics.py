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

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

sns.set_style("darkgrid")
sns.set_palette("crest_r")
sns.set(rc={"figure.dpi": 100, "savefig.dpi": 100, "figure.figsize": (12, 8)})


def _get_clustering_performance_metrics(
    df_with_clusters: pd.DataFrame, cluster_col: str, feature_blocklist: list = []
) -> dict:
    """Clustering performance.

    It takes a dataframe with a column called "cluster" and returns a dictionary with three clustering
    performance metrics

    Args:
      df_with_clusters (pd.DataFrame): data with the cluster id.
      cluster_col (str): Name of cluster columns
      feature_blocklist (list): List of features NOT to use

    Returns:
      A dictionary with the metrics as keys and the values as the values.
    """
    number_of_cluster = df_with_clusters[cluster_col].nunique()

    columns = [
        col
        for col in df_with_clusters.columns
        if (col != cluster_col and col not in feature_blocklist)
    ]
    metrics_dict = {}
    metrics_dict["sillhouette_score"] = np.nan
    metrics_dict["calinski_harabasz_score"] = np.nan
    metrics_dict["davies_bouldin_score"] = np.nan

    if number_of_cluster != 1:
        metrics_dict = {}
        metrics_dict["sillhouette_score"] = metrics.silhouette_score(
            df_with_clusters[columns],
            df_with_clusters[cluster_col],
            metric="euclidean",
            sample_size=None,
        )
        metrics_dict["calinski_harabasz_score"] = metrics.calinski_harabasz_score(
            df_with_clusters[columns], df_with_clusters[cluster_col]
        )
        metrics_dict["davies_bouldin_score"] = metrics.davies_bouldin_score(
            df_with_clusters[columns], df_with_clusters[cluster_col]
        )
    else:
        # This is the case deterministic clustering techniques
        msg1 = "One of the clustering algorithms selected 1 as the optimal number of clusters, this is non-valid cluster label."
        msg2 = (
            "\n"
            + "Filling metrics with np.nan for this experiment. You can check wich one, on the model selection report."
        )
        logger.warning(msg1 + msg2)

    return metrics_dict


def _build_metrics_ranking(
    reports_df: pd.DataFrame, metrics_computation: dict
) -> pd.DataFrame:
    """Ponderate metrics according a given weights importance.

    A higher rank score the better the cluster is.

    It takes a dataframe of reports and a dictionary of metrics and their weights, and returns a
    dataframe of reports sorted by their final score.

    Args:
      reports_df (pd.DataFrame): a dataframe containing the reports of the feature selection methods
      metrics_computation (dict): a dictionary of metrics and their weights.

    Returns:
      A dataframe with the final score for each feature selection method.
    """
    metric_weighting = metrics_computation["metric_weighting"]
    inverse_metrics_values_cols = metrics_computation["invert_metric_value"]
    metrics_list = list(metric_weighting.keys())
    normalized_metrics = [f"{col}_normalized" for col in metrics_list]
    # normalize metrics
    reports_df[normalized_metrics] = MinMaxScaler().fit_transform(
        reports_df[metric_weighting.keys()]
    )
    # for {inverse_metrics_values_cols} columns a lower value means a better clustering.
    # so we need to invert normalized value to be able to ponderate
    for col in [f"{col}_normalized" for col in inverse_metrics_values_cols]:
        reports_df[col] = 1 - reports_df[col]
    # compute final ranking
    metric_w_cols = []
    for metric in metrics_list:
        reports_df[f"{metric}_w"] = (
            reports_df[f"{metric}_normalized"] * metric_weighting[metric]
        )
        metric_w_cols.append(f"{metric}_w")
    reports_df["final_score"] = reports_df[metric_w_cols].mean(axis=1)
    reports_df = reports_df.sort_values("final_score", ascending=False).reset_index(
        drop=True
    )
    return reports_df
