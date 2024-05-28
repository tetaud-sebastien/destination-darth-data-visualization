import json
import os
import time
import pandas as pd
import numpy as np
import yaml
from loguru import logger
import earthkit.data
import earthkit.regrid
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from utils import (get_cacheB_dataset, make_predictions, plot_benchmark,
                   plot_forecast, preprocess, train_model)

def get_polytope_dataset(config: dict):

    polytope_url = config["polytope_url"]
    polytope_request = config["polytope_request"]
    data = earthkit.data.from_source("polytope", "destination-earth", polytope_request, address=polytope_url, stream=False)
    return data


def polytope_preprocess(dataset, config: dict,lat: float = 48.8566, lon: float = 2.3522,
               method: str = "nearest",
               resample_period: str = "D") -> pd.DataFrame:

    grid = config["grid"]
    out_grid = {"grid": [grid['lat'], grid['lon']]}
    # regrid healpix to lon lat
    data_latlon = earthkit.regrid.interpolate(data, out_grid=out_grid, method=grid['method'])
    # Convert to xarray
    ds = data_latlon.to_xarray()
    ds = ds["t2m"]
    # Select the nearest point
    ds = ds.sel(latitude=lat, longitude=lon, method=method)
    dataset = ds.resample(time=resample_period).mean(dim="time")
    index = dataset.time
    df = pd.DataFrame(data={"time": index,
                            "temperature": dataset.values.flatten()})
    df["temperature"] = df["temperature"] - 273

    return df


if __name__ == "__main__":
    # Dictionary to store benchmarking results
    benchmarks = {}
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Extract configuration details
    capital_coordinates = config["capital_coordinates"]
    capital_coordinates = dict(sorted(capital_coordinates.items()))
    output_folder = config["output_folder"]
    request_nb = config["request_nb"]
    # Generate list of dates for N days
    N = 10  # Number of days
    start_date = pd.to_datetime(config["polytope_request"]["date"])  # Assume the start date is provided in the config
    dates = [start_date + pd.Timedelta(days=i) for i in range(N)]


    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(dir_path, output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("start benchmark")
    # Iterate over each capital's coordinates for benchmarking
    for cap in capital_coordinates.keys():
        benchmark = {
            "access_time": [],
            "data_processing": [],
            "train_model": [],
            "model_forecast": [],
            "end_to_end": []
            }

        # Repeat benchmarking for a specified number of requests
        for _ in range(request_nb):

            coord = capital_coordinates[cap]
            logger.info(f"cap: {cap}: coord: {coord}")

            # Initialize an empty list to store datasets
            datasets = []

            for date in dates:
            # Modify the polytope_request for the current date
                config["polytope_request"]["date"] = date.strftime("%Y%m%d")
                logger.info(config["polytope_request"]["date"])

                # Query the data
                t0 = time.time()
                data = get_polytope_dataset(config=config)
                t1 = time.time()
                df_tmp = polytope_preprocess(dataset=data, config=config,
                                             lat=coord[0], lon=coord[1],
                                             method="nearest",
                                             resample_period="D")
                t2 = time.time()


                datasets.append(df_tmp)
            df = pd.concat(datasets, axis=0)
            print(df)
            df['time'] = pd.to_datetime(df['time'])

            t2 = time.time()
            model, train_df, test_df = train_model(df=df,
                                                   date_col='time',
                                                   temp_col='temperature')

            t3 = time.time()
            df_forecast, mae, rmse = make_predictions(model, test_df)
            t4 = time.time()
            plot_forecast(train_df=train_df,test_df=test_df,
                            forecast=df_forecast, city=cap,
                            coord=coord, verbose=False,
                            save=True, output_path=out_dir)
            t5 = time.time()
            # Record benchmarking times
            benchmark["access_time"].append(t1-t0)
            benchmark["data_processing"].append(t2-t1)
            benchmark["train_model"].append(t4-t3)
            benchmark["model_forecast"].append(t5-t4)
            benchmark["end_to_end"].append(t5-t0)

            #     # Calculate mean times
        benchmarks[cap] = {key: np.mean(value) for key,
                           value in benchmark.items()}
        # Calculate mean and standard deviation of times
        benchmarks[cap]["end_to_end_std"] = np.std(benchmark["end_to_end"])
        logger.warning(benchmarks)

    # Convert and write JSON object to file
    with open(os.path.join(out_dir, "benchmark.json"), "w") as outfile:
        json.dump(benchmarks, outfile)
    plot_benchmark(benchmark_dict=benchmarks,
                   out_dir=out_dir)
    logger.info("Benchmark completed. Results saved to {}", "benchmarks.json")
