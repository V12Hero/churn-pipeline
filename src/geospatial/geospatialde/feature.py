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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from operator import itemgetter
from typing import Any, Dict, List, Union

import geopandas as gpd
import mpu
import numpy as np
import pandas as pd

from segmentation_core.helpers.data_processing.general import clean_string, join_dfs
from segmentation_core.helpers.data_processing.geometry import create_buffer
from segmentation_core.helpers.data_transformers.cleaning_utils import (
    filling_nans_by_fixed_value,
)

from .functions import *  # noqa F401

logger = logging.getLogger(__name__)


def create_geometry(
    data: pd.DataFrame,
    apply: bool,
    coordinates: Dict[str, str],
    epsg: int,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Create the geometry using coordinates in the dataframe."""
    if not apply:
        return data

    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            x=data[coordinates["longitud"]], y=data[coordinates["latitud"]]
        ),
        crs=epsg,
    )


def create_features(
    data: pd.DataFrame, features: Union[Dict[str, Any], None]
) -> pd.DataFrame:
    """Create new columns using different functions."""

    if features is None:
        return data

    for col_name, params in features.items():
        if params["type"] == "function_col":
            foo = globals()[params["function"]]
            data[col_name] = foo(data, **params["parameters"])

        elif params["type"] == "function":
            foo = globals()[params["function"]]
            data = foo(data, **params["parameters"])

        else:
            raise NotImplementedError(f"{params['type']} not implemented.")

    return data


def aggregate_values(
    data: pd.DataFrame,
    groups: List[str],
    aggregation: Dict[str, List[str]],
) -> pd.DataFrame:
    """Group and aggregate the data according to the parameters."""
    if groups is None or aggregation is None:
        return data

    data = data.groupby(groups).agg(aggregation).reset_index()

    if type(data.columns) == pd.core.indexes.multi.MultiIndex:
        # Concat levels col names
        data.columns = data.columns.map("_".join).str.strip("_")

    return data


def compute_area(geometry: gpd.GeoDataFrame, epsg: int) -> pd.Series:
    """
    "Compute the area of each geometry in a GeoDataFrame."

    The first line of the docstring is a one sentence summary of the function

    Args:
      geometry (gpd.GeoDataFrame): the GeoDataFrame that you want to compute the area of.
      epsg (int): the epsg code of the CRS that the geometry is in.

    Returns:
      A pandas series with the area of each polygon in the geometry column.
    """
    geometry = geometry.to_crs(epsg)
    return geometry.area


def agg_cols(
    df: pd.DataFrame, agg_cols: List[str], agg_function: str = "sum"
) -> pd.Series:
    """
    "Applies aggregation on a given set of columns."

    Args:
      df (pd.DataFrame): input_df
      agg_cols (List[str]): List with columns to aggregate
      agg_function (str): Name of aggregate function. Defaults to 'sum'. Defaults to sum

    Returns:
      A series with the aggregated values.
    """
    return df[agg_cols].agg(agg_function, axis=1)


def compute_mean_from_bins(df: pd.DataFrame, cols_values: Dict[str, float]):
    """
    Calculates the mean value of something based on bins from a dataframe and
    a dictionary saying what mean should be considered for each bin

    Args:
      df (pd.DataFrame): the dataframe to be processed
      cols_values (Dict[str, float]): a dictionary of column names and their mean values

    Returns:
      The mean of the values in the bins.
    """
    new_df = df.copy()

    for col, mean_value in cols_values.items():
        new_df[f"{col}_tmp"] = new_df[col] * mean_value

    values = new_df[[f"{col}_tmp" for col in list(cols_values.keys())]].sum(axis=1)
    count = new_df[list(cols_values.keys())].sum(axis=1)

    return values / count


def add_percentage_cols(df: pd.DataFrame, cols: List[str]):
    """
    > It takes a dataframe and a list of columns, and returns a new dataframe with the same columns plus
    new columns that are the percentage of each column in the original list

    Example:
        >> df = pd.DataFrame({'first_col': [2], 'other_col': [3]})
        >> add_percentage_cols(df, ['first_col', 'other_col'])

        first_col  other_col  first_col_per  other_col_per
        0          2          3            0.4            0.6

    Args:
      df (pd.DataFrame): the dataframe you want to add the percentage columns to
      cols (List[str]): The columns to add percentages for

    Returns:
      A new dataframe with the percentage of each column
    """
    new_df = df.copy()
    total = new_df[cols].sum(axis=1)

    for col in cols:
        new_df[f"{col}_per"] = new_df[col] / total

    return new_df


def add_multiple_percentage_cols(df: pd.DataFrame, cols_list: List[List[str]]):
    """
    > It takes a dataframe and a list of lists of column names, and returns a new dataframe with the
    percentage columns added

    Args:
      df (pd.DataFrame): the dataframe you want to add the percentage columns to
      cols_list (List[List[str]]): a list of lists of column names. Each list of column names will be
    used to create a new column.

    Returns:
      A new dataframe with the percentage columns added.
    """
    new_df = df.copy()

    for cols in cols_list:
        new_df = add_percentage_cols(new_df, cols)

    return new_df


def time_delta(
    data: pd.DataFrame,
    start: Union[str, datetime],
    end: Union[str, datetime],
    units: str,
) -> pd.Series:
    """
    Returns the time difference between two columns on a dataframe

    Args:
      data (pd.DataFrame): The dataframe containing the start and end columns
      start (Union[str, datetime]): The start date of the time period.
      end (Union[str, datetime]): The column name or datetime object that represents the end of the time
    period.
      units (str): The units of time to return.

    Returns:
      A series of time deltas
    """
    if isinstance(start, datetime) and isinstance(end, datetime):
        raise TypeError("start and/or end must be column names")

    if isinstance(start, str):
        start = data[start]

    if isinstance(end, str):
        end = data[end]

    return (
        pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)
    ) / np.timedelta64(1, units)


def _calc_dist(new_df: pd.DataFrame) -> pd.DataFrame:
    """Compute distances between two geometry GPD series.

    > We're converting the geometry columns of the new dataframe to the same projection, then
    calculating the distance between the two geometries

    Args:
      new_df (pd.DataFrame): the dataframe that contains the geometry of the input dataframe and the
    geometry of the OSM dataframe

    Returns:
      A dataframe with the distance between the two points.
    """
    new_df["distance"] = (
        gpd.GeoSeries(new_df["geometry_osm"])
        .to_crs("epsg:3857")
        .distance(gpd.GeoSeries(new_df["geometry_input_df"]).to_crs("epsg:3857"))
    )
    return new_df


def calculate_min_mean_distance(
    df_with_distance: pd.DataFrame,
    columns: List[str],
    groupby_col: str,
    max_distance: float,
):
    """Compute Min and Mean distance to POIs.

    > For each POI, calculate the minimum and mean distance between the POI and the other POIs

    Args:
      df_with_distance (pd.DataFrame): the dataframe with the distance column
      columns (List[str]): the columns you want to calculate the distance for
      groupby_col (str): the column to group by. In this case, it's the "id" column.
      max_distance (float): the maximum distance to consider.
    """
    # distance dfs
    new_dfs = []
    filt_df = df_with_distance[(df_with_distance["distance"] != 0)]
    filt_df["tmp"] = 1
    # for each POI
    for column in columns:
        # create distance table
        df_distance = (
            filt_df.groupby([groupby_col, column])
            .agg(
                min_distance=pd.NamedAgg("distance", "min"),
                mean_distance=pd.NamedAgg("distance", "mean"),
            )
            .reset_index()
            .pivot(index=groupby_col, columns=column)
        )
        # renaming
        df_distance.columns = ["_".join(cols) for cols in df_distance.columns]
        df_distance.fillna(max_distance, inplace=True)
        # save distaces
        new_dfs.append(df_distance.reset_index())

    # join all dfs
    joined_dfs = join_dfs(groupby_col, *new_dfs)
    return joined_dfs


def combine_geospatial_features(df: pd.DataFrame, feature_dict: Dict):
    """
    > Create each feature in the feature dictionary, by aggregating the columns in the dataframe using the
    specified aggregation type

    Args:
      df (pd.DataFrame): the dataframe you want to add the features to
      feature_dict (Dict): a dictionary of features to be created. The key is the name of the feature,
    and the value is a dictionary of parameters.

    Returns:
      A dataframe with the new features added.
    """

    for feature, params in feature_dict.items():
        agg_cols = params["agg_cols"]
        agg_type = params["agg_type"]

        df[feature] = df[agg_cols].agg(agg_type, axis=1)

    return df


def _process_city(points, polygons, city_name, bbox):
    """_summary_

    Args:
        points (_type_): _description_
        polygons (_type_): _description_
        city_name (_type_): _description_
        bbox (_type_): _description_

    Returns:
        _type_: _description_
    """
    logger.info(f"Processing city {city_name}")
    logger.info(f"bbox {bbox}")
    # Your bbox better be right ;)
    polygons_city = polygons.cx[
        bbox["lon"][0] : bbox["lon"][1], bbox["lat"][0] : bbox["lat"][1]
    ]
    input_df_city = points.cx[
        bbox["lon"][0] : bbox["lon"][1], bbox["lat"][0] : bbox["lat"][1]
    ]

    polygons_city["index_polygon"] = polygons_city.index

    new_df_city = polygons_city.sjoin(input_df_city, predicate="within").rename(
        columns={"geometry": "geometry_osm", "geometry_point": "geometry_input_df"}
    )
    logger.info(new_df_city.filter(like="geometry"))
    # EPSG 3857 or pseudo-mercator projections give you an approximation to distance in meters
    new_df_city["distance"] = (
        gpd.GeoSeries(new_df_city["geometry_osm"])
        .to_crs("epsg:3857")
        .distance(gpd.GeoSeries(new_df_city["geometry_input_df"]).to_crs("epsg:3857"))
    )

    new_df_city = new_df_city.drop(["geometry_input_df", "geometry_osm"], axis=1)
    logger.info(f"Finished processing city {city_name}")
    return new_df_city


def compute_haversine_distance(
    input_df: pd.DataFrame,
    osm_df: pd.DataFrame,
    params: Dict,
    bbox: Dict,
    parallel=True,
):
    """Computes distances between POI and users.

    Calculates distances from items in input_df to osm_df in a given distance threshold.

    The function will create POI-customer distance pairs, contained within a set maximum radius.
    Heads up: When the distance between the points exceeds this radius the join gives as result 0,
    since there is no intersection between the polygons.

    Args:
        input_df (pd.DataFrame): Input dataframe
        osm_df (pd.DataFrame): OSM points dataframe
        max_diameter (float): Diameter of the city, to be used as a max distance.
        drop_na_cols: List[str]: cols to drop nulls on the OSM dataset

    Returns:
        data: (pd.DataFrame): output dataframe
    """
    max_diameter, drop_na_cols = itemgetter("max_diameter", "columns")(params)
    city_filter = params.get("city_filter", [])
    if len(city_filter) > 0:
        bbox = {city: bbox[city] for city in city_filter}

    input_df = input_df.copy()
    osm_df = osm_df.copy()
    logger.info(
        f"Dropping OSM rows that dont add information based on columns: {drop_na_cols}"
    )
    osm_df_clean = osm_df.dropna(subset=drop_na_cols)
    logger.info(
        f"Reduce data in: {(1 - (osm_df_clean.shape[0]/osm_df.shape[0]))*100} %"
    )

    # Renaming to be able to have the point information later
    logger.info("Creating distance buffer")
    input_df["geometry_point"] = input_df["geometry"]
    input_df["geometry"] = create_buffer(input_df["geometry"], distance=max_diameter)

    # Use ThreadPoolExecutor to perform spatial join on each city in parallel
    # TODO: make it work in jupyter
    if parallel:
        with ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("fork")
        ) as executor:
            futures = []
            for city_name, bbox_city in bbox.items():
                partial_process_city_i = partial(
                    _process_city,
                    points=input_df,
                    polygons=osm_df_clean,
                    city_name=city_name,
                    bbox=bbox_city,
                )
                futures.append(executor.submit(partial_process_city_i))
            results = [future.result() for future in futures]
    else:
        partial_process_city = partial(
            _process_city, points=input_df, polygons=osm_df_clean
        )
        results = []
        for city_name, bbox_city in bbox.items():
            partial_process_city.keywords["city_name"] = city_name
            partial_process_city.keywords["bbox"] = bbox_city
            results.append(partial_process_city())

    # Concatenate the results from each city
    logger.info("Concatenating results from each city")
    data = pd.concat(results, ignore_index=True)

    return data


def calculate_num_points(
    filt_df: pd.DataFrame, columns: List[str], threshold: float, groupby_col: str
):
    """
    Calculates number of interest points for each column in a given distance threshold

    Args:
        filt_df (pd.DataFrame): input Dataset
        columns (List[str]): List of columns with points
        threshold (float): Threshold distance in meters
        groupby_col (str): Column to use as reference to groupby

    Returns:
        pd.DataFrame
    """

    new_dfs = []
    filt_df["tmp"] = 1

    for column in columns:
        df_count = (
            filt_df.groupby([groupby_col, column])
            .count()
            .reset_index()
            .pivot(index=groupby_col, columns=column, values="tmp")
        )

        df_count.columns = [
            f"number_of_{col}_in_a_ratio_of_{threshold}_meters"
            for col in df_count.columns
        ]
        new_dfs.append(df_count.reset_index())

    return join_dfs(groupby_col, *new_dfs)


def compute_geographic_features(
    df_with_distance: pd.DataFrame,
    users: pd.DataFrame,
    thresholds: List[float],
    max_diameter: float,
    columns: list,
    groupby_col: str,
) -> pd.DataFrame:
    """Compute distances features.

    It takes a dataframe with distances, a list of thresholds, the diameter of the city, a list of
    columns, and a groupby column, and returns a dataframe with the number of points for different
    thresholds, the minimum and mean distance, and the diameter of the city.

    Args:
      df_with_distance (pd.DataFrame): a dataframe with the distance between each point and each POI
      users (pd.DataFrame): Users df to get the POI
      thresholds (List[float]): list of distances to calculate the number of points within
      diameter_of_the_city (float): the diameter of the city in meters. This is used to fill in the NaN values in the dataframe.
      columns (list): list of columns to group by
      groupby_col (str): the column to group by, e.g. "geohash"

    Returns:
      A dataframe with the following columns:
        - 'min_distance_to_poi'
        - 'mean_distance_to_poi'
        - 'num_points_within_X.X km'
    """
    new_dfs = []
    # df with users ids
    user_id = users[[groupby_col]]
    user_id[groupby_col] = user_id[groupby_col].astype(str)

    for threshold in thresholds:
        logger.info(f"Calculating features for threshold: {threshold}")
        # Calculates number of points for different thresholds
        filt_df = df_with_distance[
            (df_with_distance["distance"] < threshold)
            & (df_with_distance["distance"] != 0)
        ]

        if len(filt_df) == 0:
            logger.warn("Dataset empty for this threshold. Skipping feature generation")
        else:
            count_df = calculate_num_points(filt_df, columns, threshold, groupby_col)
            # fill poi counts with zeros (if there are no points in the dataset)
            count_df[groupby_col] = count_df[groupby_col].astype(str)
            count_df = count_df.merge(user_id, on=[groupby_col], how="outer")
            count_df = filling_nans_by_fixed_value(count_df, value=0)
            new_dfs.append(count_df)

    # Calculates min/mean distance only once for the largest ratio in threshold
    df_dist = calculate_min_mean_distance(
        df_with_distance, columns, groupby_col, max_diameter
    )
    df_dist[groupby_col] = df_dist[groupby_col].astype(str)
    df_dist = df_dist.merge(user_id, on=[groupby_col], how="outer")
    # fill min and mean distances with the diameter of the city
    df_dist = filling_nans_by_fixed_value(df_dist, value=max_diameter)
    new_dfs.append(df_dist)

    data = join_dfs(groupby_col, *new_dfs)
    data = filling_nans_by_fixed_value(data, value=0)

    columns = list(data.columns)
    logger.info(f"Geospatial features created: {columns}")
    return data


def filter_bbox(df: pd.DataFrame, bbox_country: Dict):
    dfs_cities = []
    for key in bbox_country.keys():
        bbox = bbox_country[key]
        points_city = df.cx[
            bbox["lon"][0] : bbox["lon"][1], bbox["lat"][0] : bbox["lat"][1]
        ]
        points_city["city"] = key
        dfs_cities.append(points_city)

    df = pd.concat(dfs_cities)

    return df


def create_country_primary(df: pd.DataFrame, country: str, bbox: Dict) -> pd.DataFrame:
    df = df.query(f"country == '{country}'")

    df = filter_bbox(df, bbox)

    # move to users prm
    df["city"] = df["city"].apply(clean_string)

    logger.info(f"Raw master table shape; {df.shape}")

    logger.info(f"{type(df)}")
    return df


def nearest_polygon_spatial_join(
    df_points: pd.DataFrame, df_poly: pd.DataFrame, distance_col: str, id_col: str
):
    # [Spatial join on meter CRS]
    result = df_points.to_crs(epsg="3857").sjoin_nearest(
        df_poly.to_crs(epsg="3857"),
        how="left",
        distance_col=distance_col,
        max_distance=1000,
    )
    # [Select smallest distance]
    result = result.sort_values(by=distance_col)
    result = result.groupby(by=id_col).head(1)

    return result


def linspace(start, stop, step):
    """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
    """
    return np.linspace(start, stop, int((stop - start) / step + 1))


def generate_primary_location_grid(params: Dict) -> pd.DataFrame:
    """Generates a grid of points based on the parameters of the bounding boxes of the cities and the resolution needed.

    Args:
        parameteres (Dict): resolution and bounding boxes

    Returns:
        pd.DataFrame: meshgrid with all the cities and countries passed through the parameters.
    """

    # read static params
    resolution = params["resolution"]
    storeCreatedAt = params["storeCreatedAt"]  # not needed TODO: remove in process
    customer_id = params["customer_id"]  # not needed TODO: remove in process

    result = [] # pd.DataFrame()

    # read list params
    countries = params["countries"]

    for country in countries:
        for city in list(params["bbox"][country].keys()):
            data = params["bbox"][country][city]

            # configure min/max lat/long
            lat_min = data["lat"][0]
            lat_max = data["lat"][1]
            long_min = data["lon"][0]
            long_max = data["lon"][1]

            # calculate distances
            dist_long = mpu.haversine_distance((lat_max, long_min), (lat_max, long_max))
            dist_lat = mpu.haversine_distance((lat_min, long_min), (lat_max, long_min))

            # calculate steps
            step_long = resolution * (long_max - long_min) / (dist_long)
            step_lat = resolution * (lat_max - lat_min) / (dist_lat)

            # generate grid
            x = linspace(lat_min, lat_max, step_lat)
            y = linspace(long_min, long_max, step_long)
            X, Y = np.meshgrid(x, y)

            # reshape grid result
            X = X.reshape((np.prod(X.shape),))
            Y = Y.reshape((np.prod(Y.shape),))

            # generate dataframe
            df = pd.DataFrame(
                {"city": city, "country": country, "latitude": X, "longitude": Y}
            )

            result.append(df)

    result = pd.concat(result)
    customer_id = linspace(customer_id, len(result), 1)
    result["customer_id"] = customer_id
    result["customer_id"] = result["customer_id"].astype(int)
    result["storeCreatedAt"] = storeCreatedAt

    return result
