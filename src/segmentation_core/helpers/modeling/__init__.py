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

"""Contains functions for running customerone_commons data transformation tasks.

This module contains functions for running data transformation
tasks including: imputation, label encoding, and outlier removal
"""

from segmentation_core.helpers.modeling.embedding import (  # noqa
    embedding_sparse_features,
)
from segmentation_core.helpers.modeling.evaluation_metrics import (  # noqa
    _build_metrics_ranking,
    _get_clustering_performance_metrics,
)
from segmentation_core.helpers.modeling.explainers import (  # noqa
    _get_model_importance,
    _plot_tree_model,
    clustering_explainer,
)
from segmentation_core.helpers.modeling.model_optimization import (  # noqa
    _check_valid_number_of_clusters,
    _clustering_optimization,
    _clustering_optimization_report,
    cluster_inference,
    train_cluster_model,
    train_cluster_with_elbow_method,
    train_clustering_wrapper,
)
