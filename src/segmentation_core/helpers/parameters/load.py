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

# from segmentation_core.config import Jinja2TemplatedConfigLoader
# from segmentation_core.definitions import ROOT_DIR

# logger = logging.getLogger(__name__)
# CONF_SOURCE = ROOT_DIR + "/conf"


# def load_parameters() -> dict:
#     """Get kedro parameters

#     Returns:
#         dict: Kedro parameters
#     """
#     # conf_loader = Jinja2TemplatedConfigLoader(
#     #     conf_source=CONF_SOURCE, globals_pattern="*globals.yml"
#     # )
#     params = conf_loader.get("parameters*", "parameters*/**", "globals*")

#     return params
