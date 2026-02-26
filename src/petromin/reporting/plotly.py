import os
import typing as tp

import pandas as pd
import plotly.express as px
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


def render_scatter_mapbox_plot(df: pd.DataFrame, args: tp.Dict[str, str]) -> px.scatter_mapbox:
    """
    Renders a clustering map using Plotly Express and Mapbox.

    Args:
        df (pd.DataFrame): The DataFrame containing data for the map.
        args (Dict[str, str]): Additional keyword arguments to customize the map.

    Returns:
        px.scatter_mapbox: A Plotly Express scatter_mapbox figure.

    Example:
        # Usage example
        map_args = {
            'lat': 'latitude',
            'lon': 'longitude',
            'zoom': 10
        }
        fig = render_scatter_mapbox_plot(data_df, map_args)
        fig.show()
    """
    # px.set_mapbox_access_token(os.environ["PLOTLY_MAPBOX_ACCESS_TOKEN"])
    # fig = px.scatter_mapbox(df, color_continuous_scale=px.colors.cyclical.IceFire, **args)
    fig = px.scatter_map(df, color_continuous_scale=px.colors.cyclical.IceFire, **args)
    return fig


def geographic_plot(df: pd.DataFrame, params: tp.Dict[str, tp.Any]) -> px.scatter_mapbox:
    """
    Generates a geographic cluster locations map and displays it.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data for the map.
        params (Dict[str, Any]): Parameters for rendering the geographic report.

    Returns:
        px.scatter_mapbox: A Plotly Express figure showing cluster locations on a map.

    Example:
        # Usage example
        map_params = {
            'geographic_report': {
                'lat': 'latitude',
                'lon': 'longitude',
                'zoom': 10
            }
        }
        fig = geographic_plot(data_df, map_params)
        fig.show()
    """
    fig = render_scatter_mapbox_plot(df, args=params["geographic_report"])
    return fig
