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
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.tree import plot_tree

from .objects.load import load_object
from .reproductibility.seed_file import get_global_seed, seed_file

logger = logging.getLogger(__name__)

sns.set_style("darkgrid")
sns.set_palette("crest_r")
sns.set(rc={"figure.dpi": 100, "savefig.dpi": 100, "figure.figsize": (12, 8)})


def clustering_explainer(
    df: pd.DataFrame,
    fs_params: Dict,
    clustering_columns: List,
    explainer_model_params: Dict,
    importance_model_params: Dict,
) -> Dict:
    """Clustering explainer using tree based algorithms.

    Explain clustering predictions using Bagging/Boosting and Decission trees.

    1. Bagging/Boosting algorithms are used to create the feature importance ranking.
    2. Tree plot is to have a visual explanation of tree predictions.

    Args:
      df (pd.DataFrame): The dataframe that contains the master table with the clustering column.
      fs_params (Dict): This is the feature selection parameters dictionary.
      clustering_columns (List): Clustering columns used to interprete model.
      explainer_model_params (Dict): Dictionary with explainer model params.
      importance_model_params (Dict): Dictionary with importance model params.

    Returns:
      A dictionary with the tree plot and feature importance dataframe
    """
    seed = get_global_seed()
    seed_file(seed, verbose=False)
    # id col & cluster cols
    id_col = fs_params["customer_id_col"]
    cluster_col = fs_params["cluster_col"]
    # target vs predictors
    predictor_names = [
        col for col in clustering_columns if col not in [cluster_col, id_col]
    ]
    # load classifier model
    feature_importance_model = load_object(importance_model_params)
    # supervised matrix and vector
    X = df[predictor_names]
    y = df[cluster_col]

    X = pd.DataFrame(X, columns=predictor_names)
    y = pd.DataFrame(y, columns=[cluster_col])
    # Fitting classifier
    feature_importance_model.fit(X, y)
    importance_df = _get_model_importance(feature_importance_model)
    # explainer model
    explainer_model = load_object(explainer_model_params)
    explainer_model.fit(X, y)
    fig_tree_plot = _plot_tree_model(explainer_model, show_fig=False)
    # output dict with tree plot and feature importance
    out_dict = {"tree_plot": fig_tree_plot, "feature_importance": importance_df}
    return out_dict


def _get_model_importance(model: sklearn.ensemble) -> pd.DataFrame:
    """Get dataframe with feature importance from an ensamble model.

    It takes a trained model and returns a dataframe with the feature importance of each feature

    Args:
      model (sklearn.ensemble): the model you want to get the feature importance from

    Returns:
      A dataframe with the feature name and the feature importance
    """
    importance = {}
    # summarize feature importance
    for i, score in enumerate(model.feature_importances_):
        importance[model.feature_names_in_[i]] = [score]
    importance_df = (
        (
            pd.DataFrame(importance)
            .transpose()
            .reset_index()
            .rename(columns={"index": "feature_name", 0: "feature_importance"})
        )
        .sort_values("feature_importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def _plot_tree_model(model: sklearn.ensemble, show_fig: bool = False) -> plt.Figure:
    """Tree model plot.

    It takes a trained tree model and plots it

    Args:
      model (sklearn.ensemble): sklearn.ensemble
      show_fig (bool): If True, will show the figure. If False, will return the figure object. Defaults
    to False

    Returns:
      A matplotlib figure object
    """
    # Building tree for explainability
    fig, ax = plt.subplots(figsize=(50, 50))
    try:
        # classifier plot
        plot_tree(
            model,
            feature_names=model.feature_names_in_,
            fontsize=10,
            filled=False,
            class_names=[str(c) for c in model.classes_],
            ax=ax,
        )
    except Exception:
        # regressor plot
        plot_tree(
            model,
            feature_names=model.feature_names_in_,
            fontsize=10,
            filled=False,
            ax=ax,
        )
    if show_fig:
        fig.show()
    return fig
