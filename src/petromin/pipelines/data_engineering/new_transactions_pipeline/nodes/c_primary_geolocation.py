"""Primary layer nodes."""

import logging
import typing as tp
from datetime import date
from math import radians
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
import pyspark.sql.dataframe
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from sklearn.neighbors import BallTree

from .c_primary_spine import create_auxillary_columns

logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()


def create_prm_geolocation(
    prm_branches: pd.DataFrame,
    geospatial_poi: gpd.GeoDataFrame,
    geospatial_highways: gpd.GeoDataFrame,
    geospatial_subways: gpd.GeoDataFrame,
    world_pop: pd.DataFrame,
) -> pyspark.sql.DataFrame:
    """Create primary geolocation table.

    This function computes and merges geolocation-related features for stores based on various data sources,
    including competitors, OSM (OpenStreetMap), and Worldpop data.

    Args:
        params: tp.Dict[str, tp.List[str]]: specific list of competitors per store type
        stores_info (pd.DataFrame): DataFrame containing store information.
        geospatial (pd.DataFrame): DataFrame containing geospatial data.
        world_pop (pd.DataFrame): DataFrame containing world population data.
        competitors (pd.DataFrame): DataFrame containing competitor information.
        spine (pyspark.sql.DataFrame): DataFrame containing spine data.

    Returns:
        pyspark.sql.DataFrame: DataFrame with computed geolocation-related features merged with spine data.

    Examples:
        Usage example of the create_prm_geolocation function:

        ```python
        from pyspark.sql import SparkSession
        import pandas as pd

        spark = SparkSession.builder.appName("pandas_to_spark").getOrCreate()

        # Sample data for stores_info DataFrame
        stores_info_data = [
            (1, 42.1234, -71.5678),
            (2, 40.9876, -73.4567),
        ]
        stores_info_columns = ['_id', 'latitude', 'longitude']
        stores_info_df = spark.createDataFrame(stores_info_data, stores_info_columns)

        # Sample data for geospatial DataFrame (as a pandas DataFrame)
        geospatial_data = pd.DataFrame({
            'latitude': [42.1234, 40.9876],
            'longitude': [-71.5678, -73.4567],
            'distance_to_point_a': [10.0, 15.0],
            'distance_to_point_b': [5.0, 12.0],
            'ratio_a_to_b': [2.0, 1.25]
        })

        # Sample data for world_pop DataFrame (as a pandas DataFrame)
        world_pop_data = pd.DataFrame({
            'latitude': [42.1234, 40.9876],
            'longitude': [-71.5678, -73.4567],
            'pop_density': [500, 750]
        })

        # Sample data for competitors DataFrame (as a pandas DataFrame)
        competitors_data = pd.DataFrame({
            'store_id': [1, 2],
            'competitor_distance': [3.0, 4.0],
            'competitor_count': [2, 3]
        })

        # Sample data for spine DataFrame (as a pyspark DataFrame)
        spine_data = [
            (1, '2023-01-01'),
            (2, '2023-01-01')
        ]
        spine_columns = ['_id', '_observ_end_dt']
        spine_df = spark.createDataFrame(spine_data, spine_columns)

        # Compute primary geolocation table
        prm_geolocation = create_prm_geolocation(
            stores_info_df,
            geospatial_data,
            world_pop_data,
            competitors_data,
            spine_df
        )
        prm_geolocation.show()
        ```
    """

    prm_branches_info = prm_branches[["branch_id", "latitude", "longitude"]].copy()

    # Stores variables
    prm_branches_info["latitude_rad"] = prm_branches_info["latitude"].astype(float).apply(radians)
    prm_branches_info["longitude_rad"] = prm_branches_info["longitude"].astype(float).apply(radians)
    prm_branches_info = prm_branches_info.dropna(axis=0, how="any")

    # OSM variables
    dataframes_list = [geospatial_poi, geospatial_highways, geospatial_subways]
    geospatial = gpd.GeoDataFrame(pd.concat(dataframes_list, ignore_index=True))
    geospatial["latitude"] = geospatial.geometry.centroid.y
    geospatial["longitude"] = geospatial.geometry.centroid.x
    geospatial["latitude_rad"] = geospatial["latitude"].astype(float).apply(radians)
    geospatial["longitude_rad"] = geospatial["longitude"].astype(float).apply(radians)

    ball = BallTree(geospatial[["latitude_rad", "longitude_rad"]].values, metric="haversine")

    radius_list = [300, 500, 1000]
    max_radius = np.max(radius_list)

    indices, distances = ball.query_radius(
        prm_branches_info[["latitude_rad", "longitude_rad"]].values, 
        r=max_radius/6371000,
        return_distance=True,
    )
    distances = distances * 6371

    prm_branches_info["geo_point_index"] = indices
    prm_branches_info["distances_osm"] = distances

    prm_branches_info = prm_branches_info.explode(["geo_point_index", "distances_osm"])

    for radius in radius_list:
        prm_branches_info[f"is_within_{radius:04}"] = np.where(
            prm_branches_info["distances_osm"] <= radius/1000,
            1,
            0
        )

    prm_branches_info = prm_branches_info.merge(
        geospatial.drop(columns=["latitude_rad", "longitude_rad", "latitude", "longitude", "geometry"]),
        right_index=True,
        left_on="geo_point_index",
        how="left",
    )

    agg_dict = {
        f"n_places_within_{radius:04}m": (f"is_within_{radius:04}", "sum")
        for radius in radius_list
    }

    ftr_geospatial_osm = prm_branches_info.groupby(
        ["branch_id", "amenity_map"],
        as_index=False
    ).agg(
        **agg_dict
    ).pivot_table(
        index="branch_id", columns="amenity_map", fill_value=0, margins=True
    )

    ftr_geospatial_osm.columns = ["__".join(col).strip() for col in ftr_geospatial_osm.columns]
    ftr_geospatial_osm = ftr_geospatial_osm.reset_index()

    # Worldpop variables
    world_pop["latitude_rad"] = world_pop["latitude"].astype(float).apply(radians)
    world_pop["longitude_rad"] = world_pop["longitude"].astype(float).apply(radians)
    ball = BallTree(
        world_pop[["latitude_rad", "longitude_rad"]].values,
        metric="haversine",
    )
    _, indices = ball.query(prm_branches_info[["latitude_rad", "longitude_rad"]].values, k=1)

    prm_branches_info["geo_point_index"] = [i[0] for i in indices]

    ftr_geospatial_worldpop = prm_branches_info.merge(
        world_pop.drop(columns=["latitude_rad", "longitude_rad", "latitude", "longitude"]),
        right_index=True,
        left_on="geo_point_index",
        how="left",
    )[["branch_id", "pop_density"]]

    ftr_geospatial = prm_branches.merge(
        ftr_geospatial_osm,
        on="branch_id",
        how="left"
    ).merge(
        ftr_geospatial_worldpop,
        on="branch_id",
        how="left"
    )

    logger.info("Transforming ftr_geospatial pandas df into a Spark DataFrame")
    # ftr_geospatial = spark.createDataFrame(ftr_geospatial)
    out = ftr_geospatial

    return out #.orderBy(["branch_id",])


def create_prm_geolocation_footprint(
    geospatial: pd.DataFrame,
    world_pop: pd.DataFrame,
):
    """Create geolocation footprint table.

    This function computes a geolocation footprint table by merging geospatial
    data with world population data based on the nearest grid points. The resulting
    DataFrame will contain computed geolocation footprint data based on the nearest grid points.

    Args:
        geospatial (pd.DataFrame): DataFrame containing geospatial data.
        world_pop (pd.DataFrame): DataFrame containing world population data.

    Returns:
        pyspark.sql.DataFrame: DataFrame with computed geolocation footprint data.

    Examples:
        Usage example of the create_prm_geolocation_footprint function:

        ```python
        from pyspark.sql import SparkSession
        import pandas as pd

        spark = SparkSession.builder.appName("pandas_to_spark").getOrCreate()

        # Sample data for geospatial DataFrame (as a pandas DataFrame)
        geospatial_data = pd.DataFrame({
            'latitude': [42.1234, 40.9876],
            'longitude': [-71.5678, -73.4567],
        })

        # Sample data for world_pop DataFrame (as a pandas DataFrame)
        world_pop_data = pd.DataFrame({
            'latitude': [42.1234, 40.9876],
            'longitude': [-71.5678, -73.4567],
            'pop_density': [500, 750]
        })

        # Compute geolocation footprint table
        geolocation_footprint = create_prm_geolocation_footprint(geospatial_data, world_pop_data)
        geolocation_footprint.show()
        ```
    """

    geospatial["latitude_rad"] = geospatial["latitude"].astype(float).apply(radians)
    geospatial["longitude_rad"] = geospatial["longitude"].astype(float).apply(radians)
    world_pop["latitude_rad"] = world_pop["latitude"].astype(float).apply(radians)
    world_pop["longitude_rad"] = world_pop["longitude"].astype(float).apply(radians)
    ball = BallTree(
        world_pop[["latitude_rad", "longitude_rad"]].values,
        metric="haversine",
    )
    _, indices = ball.query(geospatial[["latitude_rad", "longitude_rad"]].values, k=1)
    geospatial["geo_point_index"] = [i[0] for i in indices]
    out = geospatial.merge(
        world_pop.drop(columns=["latitude_rad", "longitude_rad", "latitude", "longitude"]),
        right_index=True,
        left_on="geo_point_index",
        how="left",
    )

    out = spark.createDataFrame(out)
    return out