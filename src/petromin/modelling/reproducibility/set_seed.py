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
"""Fixed seed for reproducibility."""
import logging
import random

import numpy as np
import sklearn

logger = logging.getLogger(__name__)

GLOBAL_SEED = 42


def seed_file(
    seed: int = GLOBAL_SEED,
    contain_message: bool = False,
    message="",
    verbose: bool = True,
):
    """It seeds the random number generator of the three libraries.

    Args:
      seed (int): int
      contain_message (bool): bool. Defaults to False
      message: If True, print the message.
      verbose: If True, print the message.

    Returns: None
    """
    sklearn.utils.check_random_state(seed)
    np.random.seed(seed)
    random.seed(seed)
    if verbose:
        if not contain_message:
            logger.info(f"Seeding sklearn, numpy and random libraries with the seed {seed}")
        else:
            logger.info(f"{message}")
