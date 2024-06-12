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
from IPython.display import HTML


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
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        heatmap = ax.pcolormesh(wind_speed.longitude, wind_speed.latitude, wind_speed,
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
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        heatmap = ax.pcolormesh(wind_speed.longitude, wind_speed.latitude, wind_speed.isel(time=0),
                                cmap='Blues', transform=ccrs.PlateCarree())
        cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Wind Speed [m/s]')
        ax.set_title(f'Wind Speed Animation')

        # Initialize the plot elements
        mesh = ax.pcolormesh(wind_speed.longitude, wind_speed.latitude, wind_speed.isel(time=0),
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

    def process(self, lat: float = 48.8566, lon: float = 2.3522,
               method: str = "nearest",
               resample_period: str = "D"):

        if self.format=="grib":

            ds = xr.open_dataset(self.filename, engine="cfgrib")
            # ds = ds.isel(time=0)
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
        self.fieldset = None
        self.regridded_dataset = None

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

    def select_data(self, date_range: pd.DatetimeIndex, variables=["u10","v10"]):
        """
        Selects a slice of the dataset based on the provided date range.

        Args:
            date_range (pd.DatetimeIndex): A range of dates to select from the dataset.

        Returns:
            xarray.Dataset: The selected data slice.
        """
        try:
            self.selected_data = self.dataset[variables].sel(time=date_range)
            logger.info(f"Data slice selected for date range {date_range}")
            return self.selected_data
        except Exception as e:
            logger.error(f"Error selecting data slice: {e}")
            raise

    def to_fieldset(self):
        """
        Converts the selected data slice to a Metview fieldset.

        Returns:
            Metview Fieldset: The converted fieldset.
        """
        if self.selected_data is None:
            logger.error("No data selected to convert. Call 'select_data' first.")
            raise ValueError("No data selected to convert. Call 'select_data' first.")

        try:
            self.fieldset = mv.dataset_to_fieldset(self.selected_data.squeeze())
            logger.info("Selected data slice converted to Metview fieldset successfully")
            return self.fieldset
        except Exception as e:
            logger.error(f"Error converting data slice to fieldset: {e}")
            raise

    def regrid_to_latlon(self, resolution: tuple = (0.25, 0.25)):
        """
        Regrids the fieldset to latitude/longitude coordinates at the specified resolution.

        Args:
            resolution (tuple): A tuple specifying the grid resolution (default is 0.25° x 0.25°).

        Returns:
            xarray.Dataset: The regridded dataset.
        """
        if self.fieldset is None:
            logger.error("No fieldset available. Call 'to_fieldset' first.")
            raise ValueError("No fieldset available. Call 'to_fieldset' first.")

        try:
            single_ll = mv.read(data=self.fieldset, grid=list(resolution))
            self.regridded_dataset = single_ll.to_dataset()
            logger.info(f"Fieldset regridded to latitude/longitude coordinates at resolution {resolution} successfully")
            return self.regridded_dataset
        except Exception as e:
            logger.error(f"Error regridding fieldset: {e}")
            raise

    def roll_longitude(self):
        """
        Adjusts the longitude coordinates of the regridded dataset to the [-180, 180) range and sorts.

        Returns:
            xarray.Dataset: The dataset with adjusted longitude coordinates.
        """
        if self.regridded_dataset is None:
            logger.error("No regridded dataset available. Call 'regrid_to_latlon' first.")
            raise ValueError("No regridded dataset available. Call 'regrid_to_latlon' first.")

        try:
            self.regridded_dataset = self.regridded_dataset.assign_coords(
                longitude=(((self.regridded_dataset.longitude + 180) % 360) - 180)
            ).sortby('longitude')
            logger.info("Longitude coordinates adjusted and sorted successfully")
            return self.regridded_dataset
        except Exception as e:
            logger.error(f"Error adjusting longitude coordinates: {e}")
            raise

    def calculate_wind_speed(self):
        """
        Calculates the wind speed from the regridded dataset's u and v wind components.

        Returns:
            xarray.DataArray: The computed wind speed.
        """
        if self.regridded_dataset is None:
            logger.error("No regridded dataset available. Call 'regrid_to_latlon' and 'roll_longitude' first.")
            raise ValueError("No regridded dataset available. Call 'regrid_to_latlon' and 'roll_longitude' first.")

        try:
            u10 = self.regridded_dataset.u10
            v10 = self.regridded_dataset.v10
            wind_speed = np.sqrt(u10**2 + v10**2)
            logger.info("Wind speed calculated successfully from regridded dataset")
            return wind_speed
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


