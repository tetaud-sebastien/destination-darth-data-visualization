#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module's docstring summary line.
This Python utils file contains functions for data loading, preprocessing,
visualization, modeling, and benchmarking for DestinE climate-dt.
"""
import base64
import os

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from loguru import logger
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tqdm import tqdm


def load_capitals_coordinates(file_path: str) -> dict:
    """
    Load capital coordinates from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing capital names and their coordinates.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config["capital_coordinates"]


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


def get_cacheB_dataset(url_dataset: str) -> xr.Dataset:
    """
    Download and load a dataset from a given URL.

    Args:
        url_dataset (str): URL of the dataset to be downloaded.

    Returns:
        xr.Dataset: The downloaded dataset.
    """
    data = xr.open_dataset(
        url_dataset,
        engine="zarr",
        storage_options={"client_kwargs": {"trust_env": "true"}},
        chunks={}
    )

    return data


def preprocess(dataset: xr.Dataset, lat: float = 48.8566, lon: float = 2.3522,
               method: str = "nearest",
               resample_period: str = "D") -> pd.DataFrame:
    """
    Preprocess the dataset to extract and resample temperature
    data for a specific location.

    Args:
        dataset (xr.Dataset): The input dataset containing temperature data.
        lat (float): Latitude of the location. Default is 48.8566 (Paris).
        lon (float): Longitude of the location. Default is 2.3522 (Paris).
        method (str): Method for selecting the nearest grid point. Default is "nearest".
        resample_period (str): Resampling period. Default is "D" (daily).

    Returns:
        pd.DataFrame: DataFrame containing resampled temperature data with columns 'time' and 'temperature'.
    """
    dataset = dataset.t2m
    dataset = dataset.sel(latitude=lat, longitude=lon, method=method)
    dataset = dataset.resample(time=resample_period).mean(dim="time")
    dataset = dataset.load()
    index = dataset.time

    df = pd.DataFrame(data={"time": index,
                            "temperature": dataset.values})

    df["temperature"] = df["temperature"] - 273

    return df


def basic_plot(df: pd.DataFrame, city: str, coord: str):
    """
    Plots the temperature over time from a given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'time' and 'temperature' columns.
        city (str): Name of the city.
        coord (str): Coordinates of the city.

    Returns:
        None
    """
    # Ensure the 'time' column is in datetime format
    df['time'] = pd.to_datetime(df['time'])

    # Create the plot
    plt.figure(figsize=(16, 8))
    plt.plot(df['time'], df['temperature'], color='#9999ff', label=f"{city} mean temperature")

    # Add title and labels
    plt.title(f'Daily average temperature [°C] in {city} - coordinate: {coord}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Temperature [°C]', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def train_model(df, date_col='time', temp_col='temperature'):
    """
    Prepares data and fits a Prophet model.

    Parameters:
    df (pd.DataFrame): DataFrame with date and temperature columns.
    date_col (str): Name of the date column in df.
    temp_col (str): Name of the temperature column in df.

    Returns:
    model (Prophet): Trained Prophet model.
    train_df (pd.DataFrame): Training DataFrame.
    test_df (pd.DataFrame): Testing DataFrame.
    """
    # Rename columns to fit Prophet requirements
    df.rename(columns={date_col: 'ds', temp_col: 'y'}, inplace=True)

    # Split data into train and test sets
    train_size = int(0.8 * len(df))  # 80% train, 20% test
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(train_df)

    return model, train_df, test_df


def make_predictions(model, test_df):
    """
    Makes predictions using a trained Prophet model on the test data.

    Parameters:
    model (Prophet): Trained Prophet model.
    test_df (pd.DataFrame): Testing DataFrame.

    Returns:
    forecast (pd.DataFrame): Forecast DataFrame with predictions.
    mae (float): Mean Absolute Error.
    rmse (float): Root Mean Squared Error.
    """
    # Make predictions on the test data
    forecast = model.predict(test_df)

    # Calculate MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)
    mae = mean_absolute_error(test_df['y'], forecast['yhat'])
    rmse = root_mean_squared_error(test_df['y'], forecast['yhat'])

    return forecast, mae, rmse


def plot_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  forecast: pd.DataFrame, city: str,
                  coord: str, verbose: bool = False,
                  save: bool = False,
                  output_path: str = "results"):
    """
    Plots the training data, test data, and forecast.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        forecast (pd.DataFrame): Forecast DataFrame with predictions.
        city (str): Name of the city.
        coord (str): Coordinates of the city.
        verbose (bool): Whether to display the plot. Default is False.
        save (bool): Whether to save the plot. Default is False.

    Returns:
        None
    """
    # Plot train and test data along with predictions
    plt.figure(figsize=(16, 8))
    plt.plot(train_df['ds'], train_df['y'], label='Train data', color='#9999ff')
    plt.plot(test_df['ds'], test_df['y'], label='Test data', color='#ff884d')
    plt.plot(test_df['ds'], forecast['yhat'], label='Predictions', color='#ff3333')
    plt.fill_between(test_df['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
    plt.xlabel('Date')
    plt.ylabel('Temperature [°C]')
    plt.title(f'Daily average temperature forecast in {city} - coordinate: {coord}')
    plt.legend(loc='lower right')
    plt.xticks(rotation=45)
    if verbose:
        plt.show()
    if save:
        filename = os.path.join(output_path, f"{city}.svg")
        plt.savefig(filename)
    plt.close()


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
    df = df.T
    df["City"] = df.index

    # Extract error bars
    errors = df['end_to_end_std']

    # Plotting the stacked bar chart without 'end_to_end'
    df_plot = df.drop(columns=['end_to_end', 'end_to_end_std', 'City'])
    df_plot.plot(kind='bar', stacked=True, figsize=(16, 8))

    # Overlay 'end_to_end' with error bars
    x = np.arange(len(df))
    plt.errorbar(x, df['end_to_end'], yerr=errors, fmt='o', color='black', capsize=5)

    # Set labels and title
    plt.ylabel('Time (seconds)')
    plt.title('End to End DT climate advanced benchmark')
    plt.xlabel("City")

    # Adding city names as x-tick labels
    plt.xticks(x, df['City'])

    # Display legend
    plt.legend(loc='upper right')

    # Save the figure
    filename = os.path.join(out_dir, "benchmark_barplot.svg")
    plt.savefig(filename)
    plt.show()


def create_map_with_forecast(capitals_coordinates: dict, forecast_folder: str,
                             map_center: tuple = (48.8566, 2.3522),
                             zoom: int = 6, output_file: str = 'map.html') -> folium.Map:
    """
    Creates a Folium map with markers for each capital, displaying forecast images in popups.

    Args:
        capitals_coordinates (dict): Dictionary containing capital names and their coordinates.
        forecast_folder (str): Directory where the forecast images are stored.
        map_center (tuple): Latitude and longitude of the map center. Default is (48.8566, 2.3522) (Paris).
        zoom (int): Initial zoom level for the map. Default is 6.
        output_file (str): File name to save the map as an HTML file. Default is 'map.html'.

    Returns:
        folium.Map: The generated Folium map object.
    """
    # Create a map centered at the specified location
    m = folium.Map(location=map_center, zoom_start=zoom)

    # Loop through the dictionary of capital coordinates
    for key in capitals_coordinates.keys():
        # Get the latitude and longitude of the current capital
        lat = capitals_coordinates[key][0]
        lon = capitals_coordinates[key][1]

        # Path to the plot image
        image_path = os.path.join(forecast_folder, f"{key}.svg")
        with open(image_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        # Create HTML for the popup with the plot image
        popup_html = f'<img src="data:image/svg+xml;base64,{img_base64}" alt="DT-Climate Forecast {key}" width="1000" height="500">'
        # Create a Popup with the HTML
        popup = folium.Popup(popup_html, max_width=1000)

        # Create a simple Folium icon
        icon = folium.Icon(prefix="fa", icon='circle', color='darkblue')

        # Add a marker with the popup and simple icon to the map
        folium.Marker(location=[lat, lon], popup=popup, icon=icon).add_to(m)

    # Save the map to an HTML file
    m.save(output_file)
    return m


def cacheB_forecast_map(config: dict):


    capital_coordinates = config["capital_coordinates"]
    capital_coordinates = dict(sorted(capital_coordinates.items()))
    url_dataset = config["cacheb_url"]
    output_folder = config["output_folder"]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(dir_path, output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    for (k, v) in tqdm(capital_coordinates.items(), desc="Processing requests", unit="request", ncols=100, colour="#3eedc4"):

        cap, coord  = k, v

        logger.info(f"cap: {cap}: coord: {coord}")

        try:

            dataset = get_cacheB_dataset(url_dataset=url_dataset)

        except Exception as e:
            logger.error(f"Issue in the data access or download: {e}")
            continue


        df = preprocess(dataset=dataset,
                        lat=coord[0], lon=coord[1],
                        method="nearest", resample_period="D")
        model, train_df, test_df = train_model(df=df,
                                                date_col='time',
                                                temp_col='temperature')
        df_forecast, mae, rmse = make_predictions(model, test_df)
        plot_forecast(train_df=train_df,test_df=test_df,
                        forecast=df_forecast, city=cap,
                        coord=coord, verbose=False,
                        save=True, output_path=out_dir)