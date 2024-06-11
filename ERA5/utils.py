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

        return wind_speed


def plot_benchmark(benchmark_dict: dict, out_dir: str):
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
    ax.set_title('End to End ERA5 CDS animation generation benchmark')

    # Save the figure
    filename = os.path.join(out_dir, "benchmark_barplot.png")
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