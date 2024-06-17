#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This Python utils file contains functions for data loading, preprocessing,
visualization, modeling, and benchmarking for DestinE climate-dt.
"""
import os
import cdsapi
import json
import xarray as xr
import metview as mv
import numpy as np
import yaml
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import pystac_client
import planetary_computer
from IPython.display import HTML
import re


def find_coord_name(coord_names, pattern):
    """
    Function to find coordinate names using regex

    """
    for name in coord_names:
        if pattern.search(name):
            return name
    return None


def load_config(file_path: str) -> dict:
    """
    Load YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing configuration information.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def calculate_wind_speed(u10, v10):
    return np.sqrt(u10**2 + v10**2)


class WindSpeedVisualizer:


    @staticmethod
    def plot_wind_speed(wind_speed):
        """
        Plot wind speed data on a map.
        """
        lat_pattern = re.compile(r'lat(itude)?', re.IGNORECASE)
        lon_pattern = re.compile(r'lon(gitude)?', re.IGNORECASE)
        coord_names = wind_speed.coords.keys()
        # Detect latitude, longitude, and time coordinates
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        heatmap = ax.pcolormesh(wind_speed[lon_name], wind_speed[lat_name], wind_speed,
                                cmap='Blues', transform=ccrs.PlateCarree())
        cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Wind Speed [m/s]')
        plt.title(f'Wind Speed on {np.datetime_as_string(wind_speed.time.values, unit="D")}', fontsize=16)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    @staticmethod
    def generate_animation(wind_speed):
        """
        Generate an animation of wind speed data.
        """
        lat_pattern = re.compile(r'lat(itude)?', re.IGNORECASE)
        lon_pattern = re.compile(r'lon(gitude)?', re.IGNORECASE)
        coord_names = wind_speed.coords.keys()
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        # Detect latitude, longitude, and time coordinates
        lat_name = find_coord_name(coord_names, lat_pattern)
        lon_name = find_coord_name(coord_names, lon_pattern)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        heatmap = ax.pcolormesh(wind_speed[lon_name], wind_speed[lat_name], wind_speed.isel(time=0),
                                cmap='Blues', transform=ccrs.PlateCarree())
        cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Wind Speed [m/s]')
        ax.set_title(f'Wind Speed Animation')

        # Initialize the plot elements
        mesh = ax.pcolormesh(wind_speed[lon_name], wind_speed[lat_name], wind_speed.isel(time=0),
                            cmap='Blues', transform=ccrs.PlateCarree())

        # Function to update the plot for each frame of the animation
        def update(frame):
            # Update the properties of the existing plot elements
            mesh.set_array(wind_speed.isel(time=frame).values.flatten())
            ax.set_title(f'Wind Speed on {np.datetime_as_string(wind_speed.time[frame].values, unit="D")}')

            return mesh,

        # Create the animation
        animation = FuncAnimation(fig, update, frames=len(wind_speed.time), interval=200, blit=True)

        # Display the animation
        plt.close()  # Close initial plot to prevent duplicate display
        return HTML(animation.to_html5_video())


class CdsERA5:

    def __init__(self):
        """
        """
        try:
            self.client = cdsapi.Client()
            logger.info("Successfully log to Climate Data Store")
        except:
            logger.error("Could not log to Climate Data Store")

    def get_data(self, query):
        """
        """
        name = query["name"]
        request = query["request"]
        self.format = query["request"]["format"]
        self.result = self.client.retrieve(name, request)
        return self.result

    def download(self, filename):
        """
        """
        self.filename = f"{filename}.{self.format}"
        self.result.download(self.filename)

    def process(self):

        if self.format=="grib":

            ds = xr.open_dataset(self.filename, engine="cfgrib")
            wind_speed = calculate_wind_speed(ds.u10, ds.v10)

        return wind_speed, ds


class GcpERA5:
    def __init__(self, zarr_path: str):
        """
        Initializes the ERA5Processor class and loads the ERA5 reanalysis data from the specified Zarr path.

        Args:
            zarr_path (str): The path to the Zarr store containing ERA5 reanalysis data.
        """
        self.zarr_path = zarr_path
        self.dataset = None
        self.selected_data = None

        try:
            self.dataset = xr.open_zarr(
                self.zarr_path,
                chunks={'time': 48},
                consolidated=True,
            )
            logger.info(f"ERA5 reanalysis data loaded successfully from {self.zarr_path}")
        except Exception as e:
            logger.error(f"Error loading ERA5 data from Zarr store: {e}")
            raise

    def get_data(self, date_range: pd.DatetimeIndex,
                    variables=["10m_u_component_of_wind",
                               "10m_v_component_of_wind"]):
        """
        Selects a slice of the dataset based on the provided date range.

        Args:
            date_range (pd.DatetimeIndex): A range of dates to select from the dataset.

        Returns:
            xarray.Dataset: The selected data slice.
        """
        try:
            self.selected_data = self.dataset[variables].sel(time=date_range)
            logger.info(f"Data slice selected for date range")
            return self.selected_data
        except Exception as e:
            logger.error(f"Error selecting data slice: {e}")
            raise

    def download(self):
        """
        """
        self.selected_data = self.selected_data.load()


    def calculate_wind_speed(self):
        """
        Calculates the wind speed from the regridded dataset's u and v wind components.

        Returns:
            xarray.DataArray: The computed wind speed.
        """

        try:
            u10 = self.selected_data["10m_u_component_of_wind"]
            v10 = self.selected_data["10m_v_component_of_wind"]
            wind_speed = np.sqrt(u10**2 + v10**2)
            logger.info("Wind speed calculated successfully from regridded dataset")
            return wind_speed, self.selected_data
        except Exception as e:
            logger.error(f"Error calculating wind speed: {e}")
            raise


class PlanetaryComputerERA5:
    def __init__(self, stac_url: str):
        """
        Initializes the ERA5Processor class and loads the ERA5 reanalysis data from the specified Zarr path.

        Args:
            stac_url (str): The path to the Zarr store containing ERA5 reanalysis data.
        """
        self.stac_url = stac_url
        self.catalog = None
        self.selected_data = None

        try:
            self.catalog = pystac_client.Client.open(self.stac_url)

            logger.info(f"Access to ERA5 reanalysis successfully from {self.stac_url}")
        except Exception as e:
            logger.error(f"Error Access ERA5 data from Url store: {e}")
            raise

    def get_data(self, date_range: pd.DatetimeIndex,
                    variables=["northward_wind_at_10_metres",
                               "eastward_wind_at_10_metres"]):

        try:
            search = self.catalog.search(collections=["era5-pds"], datetime=date_range[0], query={"era5:kind": {"eq": "an"}})
            items = search.get_all_items()
            signed_item = planetary_computer.sign(items[0])
            self.datasets = [xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
                for asset in signed_item.assets.values()]
            self.selected_data = xr.combine_by_coords(self.datasets, join="exact")
            self.selected_data  = self.selected_data[variables].sel(time=date_range)
            return
        except Exception as e:
            logger.error(f"Error selecting data slice: {e}")
            raise

    def download(self):
        """
        """
        self.selected_data = self.selected_data.load()


    def calculate_wind_speed(self):
        """
        Calculates the wind speed from the regridded dataset's u and v wind components.

        Returns:
            xarray.DataArray: The computed wind speed.
        """

        try:
            u10 = self.selected_data["northward_wind_at_10_metres"]
            v10 = self.selected_data["eastward_wind_at_10_metres"]
            wind_speed = np.sqrt(u10**2 + v10**2)
            logger.info("Wind speed calculated successfully from regridded dataset")
            return wind_speed, self.selected_data
        except Exception as e:
            logger.error(f"Error calculating wind speed: {e}")
            raise


def plot_benchmark(benchmark_dict: dict, out_dir: str, title: str):
    """
    Plot benchmark results as a stacked bar chart with error bars.

    Parameters:
        benchmark_dict (dict): Dictionary containing benchmark results.
        out_dir (str): Output directory to save the plot.

    Returns:
        None
    """
    # Convert benchmark dictionary to DataFrame
    df = pd.DataFrame(benchmark_dict)
    df = df.drop(columns=['request_issues'])

    # Calculate average and standard deviation
    means = df.mean()
    errors = df.std()
    # Plotting the stacked bar chart
    ax = means.plot(kind='bar', stacked=True, figsize=(16, 8), yerr=errors, capsize=5)

    # Set labels and title
    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Benchmark steps')
    ax.set_title(title)

    # Save the figure
    filename = os.path.join(out_dir, "benchmark_barplot.svg")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def save_results(data: dict, filename: str):
    """
    Save a dictionary to a JSON file.

    Parameters:
        data (dict): Dictionary to be saved.
        filename (str): Name of the JSON file to save.

    Returns:
        None
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


