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
from tqdm import tqdm


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
    df['time'] = pd.to_datetime(df['time'])


    return df


def load_conf(path):

    with open(path) as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":


    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = load_conf(path=os.path.join(dir_path,"config.yaml"))

    # Dictionary to store benchmarking results
    benchmarks = {}
    # Extract configuration details
    capital_coordinates = config["capital_coordinates"]
    capital_coordinates = dict(sorted(capital_coordinates.items()))
    output_folder = config["output_folder"]
    num_requests = config["num_requests"]
    # Generate list of dates for N days
    start_date = pd.to_datetime(str(config["start_date"]))
    end_date = pd.to_datetime(str(config["end_date"]))
    freq = config["freq"]
    # Create the date range with a monthly step
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    out_dir = os.path.join(dir_path, output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger.info("start benchmark")
    # Iterate over each capital's coordinates for benchmarking
    for cap in capital_coordinates.keys():

        benchmark = {
            "download_time": [None]* num_requests,
            "data_processing": [None]* num_requests,
            "train_model": [None]* num_requests,
            "model_forecast": [None]* num_requests,
            "end_to_end": [None]* num_requests,
            "request_issues": [None]* num_requests,
            }
        request_issues = 0
        # Repeat benchmarking for a specified number of requests
        for r in tqdm(range(num_requests), desc="Processing requests", unit="request", ncols=100, colour="#3eedc4"):

            coord = capital_coordinates[cap]
            logger.info(f"cap:{cap}: coord:{coord}")

            # Initialize an empty list to datasets grib file path
            datasets = []
            try:
                for i in range(len(date_range)-1):

                    # Modify the polytope_request for the current date
                    start_date = date_range[i].strftime("%Y%m%d")
                    end_date = date_range[i+1].strftime("%Y%m%d")
                    date_conf = f"{start_date}/to/{end_date}"
                    # config["polytope_request"]["date"] = date.strftime("%Y%m%d")
                    config["polytope_request"]["date"] = date_conf
                    logger.warning(config["polytope_request"]["date"])
                    # Query the data
                    t0 = time.time()
                    data = get_polytope_dataset(config=config)
                    t1 = time.time()
                    datasets.append(data)

            except Exception as e:
                logger.error(f"Issue in the data access or download: {e}")
                request_issues += 1
                continue  # Skip the current iteration and move to the next one

            df_list = []
            for data in datasets:

                df_tmp = polytope_preprocess(dataset=data, config=config,
                                                    lat=coord[0], lon=coord[1],
                                                    method="nearest",
                                                    resample_period="D")
                df_list.append(df_tmp)
            if len(df_list)>0:
                df = pd.concat(df_list, axis=0)
            else:
                df = df_list[0]

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
            benchmark["download_time"][r]=(t1-t0)
            benchmark["data_processing"][r]=(t2-t1)
            benchmark["train_model"][r]=(t4-t3)
            benchmark["model_forecast"][r]=(t5-t4)
            benchmark["end_to_end"][r]=(t5-t0)
            benchmark["request_issues"][r] = request_issues


        logger.error(benchmark)


        try:
        # Calculate mean times
            benchmarks[cap] = {key: np.mean(value) for key,
                            value in benchmark.items()}
        except Exception as e:
            continue

        # benchmarks[cap] = {key: np.mean([item for item in value if item is not None])for key,
        #                    value in benchmark.items()}
        # Calculate mean and standard deviation of times
        benchmarks[cap]["end_to_end_std"] = np.std(benchmark["end_to_end"])

        logger.warning(benchmarks)

    # Convert and write JSON object to file
    with open(os.path.join(out_dir, "benchmark.json"), "w") as outfile:
        json.dump(benchmarks, outfile)
    plot_benchmark(benchmark_dict=benchmarks,
                   out_dir=out_dir)
    logger.info("Benchmark completed. Results saved to {}", "benchmarks.json")
