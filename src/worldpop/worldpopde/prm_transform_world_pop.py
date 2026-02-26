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

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from .geoindex import get_h3_index


def transform_urbanicity_primary(int_world_pop: DataFrame) -> DataFrame:
    return (
        int_world_pop.select(
            "num_people_per_1sqkm",
            get_h3_index(F.col("lat_pop"), F.col("lon_pop"), F.lit(8)).alias(
                "h3_index_08"
            ),
        )
        .groupby("h3_index_08")
        .agg(F.avg("num_people_per_1sqkm").alias("people_density"))
    )
