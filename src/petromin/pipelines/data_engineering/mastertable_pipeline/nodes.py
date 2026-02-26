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

from typing import Optional

import geopandas as gpd
import pandas as pd


def join_pipelines_outputs(
    geospatial: pd.DataFrame,
    params: dict,
    census: Optional[pd.DataFrame] = None,
    worldpop: Optional[pd.DataFrame] = None,
    transactions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge the provided dataframes into a single dataframe based on specified merge types.

    Args:
        geospatial (pd.DataFrame, optional): Geospatial dataframe (This input dataframe should not be missing).
        census (pd.DataFrame, optional): Census dataframe.
        worldpop (pd.DataFrame, optional): World population dataframe.
        transactions (pd.DataFrame, optional): Transactions dataframe.

    Returns:
        pd.DataFrame: Merged dataframe.

    Notes:
        - The merge is performed sequentially based on the provided dataframes.
        - The merge types used depend on the presence or absence of each dataframe:
            - If geospatial is missing, an empty dataframe is returned.
            - If census is provided, it is merged with geospatial using an inner merge.
            - If worldpop is provided, it is merged with the previous result using an outer merge.
            - If transactions is provided, it is merged with the previous result using a left merge.

    Examples:
        # Example usage
        geospatial = pd.DataFrame({'ID': [1, 2, 3], 'Location': ['A', 'B', 'C']})
        census = pd.DataFrame({'ID': [2, 3, 4], 'Population': [100, 200, 300]})
        worldpop = pd.DataFrame({'ID': [1, 2, 3, 5], 'WorldPopulation': [1000, 2000, 3000, 4000]})
        transactions = pd.DataFrame({'ID': [3, 4, 5], 'Amount': [10, 20, 30]})

        merged = merge_dataframes(geospatial, census, worldpop, transactions)
        print(merged)
    """

    longitude = params["longitude"]
    latitude = params["latitude"]
    merge_method = params["merge_method"]
    customer_id = params["customer_id"]
    geometry = params["point_column_name"]

    geospatial[geometry] = gpd.points_from_xy(
        geospatial[longitude], geospatial[latitude]
    )

    geospatial[customer_id] = geospatial[customer_id].astype(int)

    if geospatial is None:
        return pd.DataFrame()

    # Merge the geospatial dataframe with the census dataframe
    if census is not None:
        merged_df = geospatial.sjoin_nearest(census, how=merge_method)
        merged_df = merged_df.drop(["index_right"], axis=1)
    else:
        merged_df = geospatial

    # Merge the worldpop dataframe with the merged_df
    if worldpop is not None:
        worldpop[geometry] = gpd.points_from_xy(worldpop[longitude], worldpop[latitude])
        worldpop = gpd.GeoDataFrame(worldpop, crs="EPSG:4326", geometry=geometry)
        merged_df = merged_df.sjoin_nearest(
            worldpop[[geometry, "fea_people_density_sqkm"]], how=merge_method
        )
        merged_df = merged_df.drop(["index_right"], axis=1)

    # Merge the transactions dataframe with the merged_df
    if transactions is not None:
        merged_df = pd.merge(
            merged_df,
            transactions,
            how=merge_method,
            on=[customer_id],  # TODO: spatial join with id or something similar
        )

    merged_df = merged_df.drop(columns=[geometry])

    feature_store_xlsx = merged_df.copy()

    return merged_df, feature_store_xlsx
