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

from segmentation_core.helpers.reporting_html.aux_report import (  # noqa
    create_gaps,
    create_mean_model,
    create_shift_model,
    display_baseline_models,
    mape,
    plot_heatmap,
    plotly_hist,
    plotly_pred_act,
    plotly_scatter,
    shap_importance,
    show_perf_metrics,
)
from segmentation_core.helpers.reporting_html.reporting import (  # noqa
    _create_folders,
    _run_template,
    create_html_report,
    create_ipynb_report,
)
from segmentation_core.helpers.reporting_html.reporting_utils import (  # noqa
    load_context,
    mprint,
    set_env_var,
)
