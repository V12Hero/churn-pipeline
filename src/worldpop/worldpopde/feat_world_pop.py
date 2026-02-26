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

from typing import Dict

import geopandas as gpd
import pandas as pd
import pyspark.pandas as ps
import pyspark.sql as spark
import pyspark.sql.functions as F

from .geoindex import get_h3_index, get_h3_neighbours


def geopandas2spark(
    prm_maestro_cliente: gpd.GeoDataFrame, cols: Dict
) -> spark.DataFrame:
    prm_maestro_cliente = prm_maestro_cliente[list(cols.values())]

    prm_maestro_cliente = ps.from_pandas(pd.DataFrame(prm_maestro_cliente)).to_spark()

    return prm_maestro_cliente


def pandas2spark(meshgrid: pd.DataFrame, cols: Dict) -> spark.DataFrame:
    meshgrid = meshgrid[list(cols.values())]

    meshgrid = ps.from_pandas(meshgrid).to_spark()

    return meshgrid


def create_features_urbanicity(
    prm_urbanicidad: spark.DataFrame, prm_maestro_cliente: spark.DataFrame, cols: Dict
) -> spark.DataFrame:
    prm_clientes = prm_maestro_cliente.select(
        cols["customer_id"],
        cols["lat"],
        cols["lon"],
        get_h3_index(F.col(cols["lat"]), F.col(cols["lon"]), F.lit(8)).alias(
            "h3_index_08"
        ),
    ).where((F.col(cols["lat"]).isNotNull()) & (F.col(cols["lon"]).isNotNull()))

    prm_neighbour_indexes = (
        prm_clientes.select(
            cols["customer_id"],
            F.explode(get_h3_neighbours(F.col("h3_index_08"), F.lit(1))).alias(
                "h3_index_08"
            ),
        )
        .join(prm_urbanicidad, how="left", on="h3_index_08")
        .groupBy(cols["customer_id"])
        .agg(F.avg("people_density").alias("people_density_neighbours"))
    )
    fea_cliente_urbanicidad = (
        prm_clientes.join(prm_urbanicidad, how="left", on="h3_index_08")
        .join(prm_neighbour_indexes, how="left", on=cols["customer_id"])
        .select(
            cols["customer_id"],
            cols["lat"],
            cols["lon"],
            F.coalesce(
                F.col("people_density"), F.col("people_density_neighbours"), F.lit(0.0)
            ).alias("fea_people_density_sqkm"),
        )
        .where(F.col("fea_people_density_sqkm").isNotNull())
    )
    return fea_cliente_urbanicidad.toPandas()
