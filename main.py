from utils import *
from loguru import logger
import time
import json
import os

# Define the coordinates dictionary
capitals_coordinates = {
    "Tirana": (41.3275, 19.8189),
    "Andorra la Vella": (42.5078, 1.5211),
    # "Yerevan": (40.1792, 44.4991),
    # "Vienna": (48.2082, 16.3738),
    # "Baku": (40.4093, 49.8671),
    # "Minsk": (53.9045, 27.5615),
    # "Brussels": (50.8503, 4.3517),
    # "Sarajevo": (43.8563, 18.4131),
    # "Sofia": (42.6977, 23.3219),
    # "Zagreb": (45.8150, 15.9819),
    # "Nicosia": (35.1856, 33.3823),
    # "Prague": (50.0755, 14.4378),
    # "Copenhagen": (55.6761, 12.5683),
    # "Tallinn": (59.4370, 24.7535),
    # "Helsinki": (60.1695, 24.9355),
    # "Paris": (48.8566, 2.3522),
    # "Tbilisi": (41.7151, 44.8271),
    # "Berlin": (52.5200, 13.4050),
    # "Athens": (37.9838, 23.7275),
    # "Budapest": (47.4979, 19.0402),
    # "Reykjavik": (64.1355, -21.8954),
    # "Dublin": (53.3498, -6.2603),
    # "Rome": (41.9028, 12.4964),
    # "Nur-Sultan": (51.1694, 71.4491),
    # "Pristina": (42.6629, 21.1655),
    # "Riga": (56.9496, 24.1052),
    # "Vaduz": (47.1410, 9.5209),
    # "Vilnius": (54.6872, 25.2797),
    # "Luxembourg City": (49.6117, 6.1319),
    # "Valletta": (35.8989, 14.5146),
    # "Chișinău": (47.0105, 28.8638),
    # "Monaco": (43.7384, 7.4246),
    # "Podgorica": (42.4418, 19.2663),
    # "Amsterdam": (52.3676, 4.9041),
    # "Skopje": (41.9973, 21.4280),
    # "Oslo": (59.9139, 10.7522),
    # "Warsaw": (52.2297, 21.0122),
    # "Lisbon": (38.7223, -9.1393),
    # "Bucharest": (44.4268, 26.1025),
    # "Moscow": (55.7558, 37.6173),
    # "San Marino": (43.9333, 12.4500),
    # "Belgrade": (44.7866, 20.4489),
    # "Bratislava": (48.1486, 17.1077),
    # "Ljubljana": (46.0569, 14.5058),
    # "Madrid": (40.4168, -3.7038),
    # "Stockholm": (59.3293, 18.0686),
    # "Bern": (46.9481, 7.4474),
    # "Ankara": (39.9334, 32.8597),
    # "Kyiv": (50.4501, 30.5234),
    # "London": (51.5074, -0.1278),
    # "Vatican City": (41.9029, 12.4534)
}

if __name__ == "__main__":
    
    benchmark_results = {}
    URL_DATASET = "https://cacheb.dcms.e2e.desp.space/destine-climate-dt/SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr"
    logger.info("start benchmark")

    for cap in capitals_coordinates.keys():
        
        coord = capitals_coordinates[cap]
        logger.info(f"cap: {cap}: coord: {coord}")
        t0 = time.time()
        dataset = get_cacheB_dataset(url_dataset=URL_DATASET)
        t1 = time.time()
        df = preprocess(dataset, lat=coord[0], lon=coord[1], method="nearest", resample_period="7D")
        t2 = time.time()    
        model, train_df, test_df = train_model(df, date_col='time', temp_col='temperature')
        t3 = time.time()
        df_forecast, mae, rmse = make_predictions(model, test_df)
        t4 = time.time()
        plot_forecast(train_df=train_df, 
              test_df=test_df, 
              forecast=df_forecast,
              city=cap, 
              coord=coord,
              verbose = False)
        t5 = time.time()

        benchmark_results[cap] = {"access_time":t1-t0,
                                  "data_processing":t2-t1,
                                  "train_model":t4-t3,
                                  "model_forecast":t5-t4,
                                  "end_to_end": t5-t0}

    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(dir_path,"results")
    # Convert and write JSON object to file
    with open(os.path.join(out_dir,"benchmark_results.json", "w")) as outfile: 
        json.dump(benchmark_results, outfile)
    plot_benchmark(benchmark_dict=benchmark_results,
                   out_dir=out_dir)
    logger.info("Benchmark completed. Results saved to {}", "benchmark_results.json")
        

        
        
    
    
   