import json
import os
import time

import numpy as np
import yaml
from loguru import logger

from utils import (get_cacheB_dataset, make_predictions, plot_benchmark,
                   plot_forecast, preprocess, train_model)

if __name__ == "__main__":
    # Dictionary to store benchmarking results
    benchmarks = {}
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Extract configuration details
    capital_coordinates = config["capital_coordinates"]
    capital_coordinates = dict(sorted(capital_coordinates.items()))
    url_dataset = config["cacheb_url"]
    output_folder = config["output_folder"]
    request_nb = config["request_nb"]

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
            t0 = time.time()
            dataset = get_cacheB_dataset(url_dataset=url_dataset)
            t1 = time.time()
            df = preprocess(dataset=dataset,
                            lat=coord[0], lon=coord[1],
                            method="nearest", resample_period="D")
            t2 = time.time()
            model, train_df, test_df = train_model(df=df,
                                                   date_col='time',
                                                   temp_col='temperature')
            t3 = time.time()
            df_forecast, mae, rmse = make_predictions(model, test_df)
            t4 = time.time()
            plot_forecast(train_df=train_df,
                          test_df=test_df,
                          forecast=df_forecast,
                          city=cap,
                          coord=coord,
                          verbose=False,
                          save=True)
            t5 = time.time()
            # Record benchmarking times
            benchmark["access_time"].append(t1-t0)
            benchmark["data_processing"].append(t2-t1)
            benchmark["train_model"].append(t4-t3)
            benchmark["model_forecast"].append(t5-t4)
            benchmark["end_to_end"].append(t5-t0)

            logger.warning(benchmark)

        # Calculate mean times
        benchmarks[cap] = {key: np.mean(value) for key,
                           value in benchmark.items()}
        # Calculate mean and standard deviation of times
        benchmarks[cap]["end_to_end_std"] = np.std(benchmark["end_to_end"])
        logger.warning(benchmarks)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(dir_path, output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Convert and write JSON object to file
    with open(os.path.join(out_dir, "benchmark.json"), "w") as outfile:
        json.dump(benchmarks, outfile)
    plot_benchmark(benchmark_dict=benchmarks,
                   out_dir=out_dir)
    logger.info("Benchmark completed. Results saved to {}", "benchmarks.json")
