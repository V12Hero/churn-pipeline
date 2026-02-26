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
import time

import requests
from tqdm import tqdm as tqdm

logger = logging.getLogger(__name__)


def get_write_path(url: str, path: str):
    """Get save path for an specific file."""
    filename = url.split("/")[-1]
    write_path = path + "/" + filename

    return write_path


def download(url: str, path: str):
    """Download util with loading bar and wait time."""
    time.sleep(5)
    chunk_size = 1024

    logger.info(f"Downloading from {url}")
    r = requests.get(url, stream=True)

    write_path = get_write_path(url, path)
    logger.info(f"Writing to {write_path}")
    total_size = int(r.headers["content-length"])
    with open(write_path, "wb") as f:
        for data in tqdm(
            iterable=r.iter_content(chunk_size=chunk_size),
            total=total_size / chunk_size,
            unit="KB",
        ):
            f.write(data)

    return write_path
