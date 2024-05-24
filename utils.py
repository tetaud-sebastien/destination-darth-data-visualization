import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os 


capitals_coordinates = {
    "Vienna": (48.2082, 16.3738),
    "Brussels": (50.8503, 4.3517),
    "Sofia": (42.6977, 23.3219),
    "Zagreb": (45.8150, 15.9819),
    "Nicosia": (35.1856, 33.3823),
    "Prague": (50.0755, 14.4378),
    "Copenhagen": (55.6761, 12.5683),
    "Tallinn": (59.4370, 24.7535),
    "Helsinki": (60.1695, 24.9355),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Athens": (37.9838, 23.7275),
    "Budapest": (47.4979, 19.0402),
    "Dublin": (53.3498, -6.2603),
    "Rome": (41.9028, 12.4964),
    "Riga": (56.9496, 24.1052),
    "Vilnius": (54.6872, 25.2797),
    "Luxembourg": (49.6117, 6.1319),
    "Valletta": (35.8970, 14.5126),
    "Amsterdam": (52.3676, 4.9041),
    "Warsaw": (52.2297, 21.0122),
    "Lisbon": (38.7223, -9.1393),
    "Bucharest": (44.4268, 26.1025),
    "Bratislava": (48.1486, 17.1077),
    "Ljubljana": (46.0569, 14.5058),
    "Madrid": (40.4168, -3.7038),
    "Stockholm": (59.3293, 18.0686)
}

def display_coordinates(city):
    
    global selected_coordinates
    global selected_city
    coords = capitals_coordinates[city]
    selected_coordinates = coords
    selected_city = city
    # print(f"Coordinates of {city}: Latitude = {coords[0]}, Longitude = {coords[1]}")
    return selected_coordinates, city

def get_cacheB_dataset(url_dataset):
           
        data = xr.open_dataset(
        url_dataset,
        engine="zarr",
        storage_options={"client_kwargs": {"trust_env": "true"}},
        chunks={})
        
        return data

    
def preprocess(dataset, lat=48.8566, lon=2.3522, city="Paris", method="nearest", resample_period="D"):
    
    """
    """
    dataset = dataset.t2m
    dataset = dataset.sel(latitude=lat, longitude=lon, method=method)
    dataset = dataset.resample(time=resample_period).mean(dim="time")
    dataset = dataset.load()
    index = dataset.time
    
    df = pd.DataFrame(data={"time":index,
                            "temperature": dataset.values})

    df["temperature"] = df["temperature"] - 273
    
    return df

    
def basic_plot(df, city, coord, verbose=False):
        """
        Plots temperature over time from a given dataframe.

        Parameters:
        df (pd.DataFrame): DataFrame with 'time' and 'temperature' columns.
        """
       
        # Ensure the 'time' column is in datetime format
        df['time'] = pd.to_datetime(df['time'])
        # Create the plot
        plt.figure(figsize=(16, 8))
        plt.plot(df['time'], df['temperature'], color='#9999ff', label=f"{city} mean temperature")
        # Add title and labels
        plt.title(f'Daily average temperature [°C] in {city} - coordinate:{coord}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Temperature [°C]', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend()
        if verbose:
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


def plot_forecast(train_df, test_df, forecast, city, coord, verbose=False, save=False):
    """
    Plots the training data, test data, and forecast.

    Parameters:
    train_df (pd.DataFrame): Training DataFrame.
    test_df (pd.DataFrame): Testing DataFrame.
    forecast (pd.DataFrame): Forecast DataFrame with predictions.
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
        
        filaname = os.path.join("results",f"{city}.svg")
        plt.savefig(filaname)
    plt.close()

        
def plot_benchmark(benchmark_dict: dict, out_dir: str):
    
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
    plt.savefig("results/Benchmark_standard_barplot_v2.svg")
        
    
    
    