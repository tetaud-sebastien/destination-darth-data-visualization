from utils import *
from loguru import logger
import time
import json
import os
import yaml


# Define the coordinates dictionary
# capital_coordinates = {
#     "Vienna": (48.2082, 16.3738),
#     "Brussels": (50.8503, 4.3517),
#     "Sofia": (42.6977, 23.3219),
#     "Zagreb": (45.8150, 15.9819),
#     "Nicosia": (35.1856, 33.3823),
#     "Prague": (50.0755, 14.4378),
#     "Copenhagen": (55.6761, 12.5683),
#     "Tallinn": (59.4370, 24.7535),
#     "Helsinki": (60.1695, 24.9355),
#     "Paris": (48.8566, 2.3522),
#     "Berlin": (52.5200, 13.4050),
#     "Athens": (37.9838, 23.7275),
#     "Budapest": (47.4979, 19.0402),
#     "Dublin": (53.3498, -6.2603),
#     "Rome": (41.9028, 12.4964),
#     "Riga": (56.9496, 24.1052),
#     "Vilnius": (54.6872, 25.2797),
#     "Luxembourg": (49.6117, 6.1319),
#     "Valletta": (35.8970, 14.5126),
#     "Amsterdam": (52.3676, 4.9041),
#     "Warsaw": (52.2297, 21.0122),
#     "Lisbon": (38.7223, -9.1393),
#     "Bucharest": (44.4268, 26.1025),
#     "Bratislava": (48.1486, 17.1077),
#     "Ljubljana": (46.0569, 14.5058),
#     "Madrid": (40.4168, -3.7038),
#     "Stockholm": (59.3293, 18.0686)
# }


if __name__ == "__main__":
    
    benchmark_results = {}
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    
    capital_coordinates = config["capital_coordinates"]  
    capital_coordinates = dict(sorted(capital_coordinates.items()))
    url_dataset = config["cacheb_url"]
    output_folder = config["output_folder"]
    request_nb = config["request_nb"]
    
    logger.info("start benchmark")
    for cap in capital_coordinates.keys():   
        benchmark_result = {
            "access_time": [],
            "data_processing": [],
            "train_model": [],
            "model_forecast": [],
            "end_to_end": []
            }
        
        for _ in range(request_nb):

            coord = capital_coordinates[cap]
            logger.info(f"cap: {cap}: coord: {coord}")
            t0 = time.time()
            dataset = get_cacheB_dataset(url_dataset=url_dataset)
            t1 = time.time()
            df = preprocess(dataset, lat=coord[0], lon=coord[1], method="nearest", resample_period="D")
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
                verbose = False,
                save=True)
            t5 = time.time()

            benchmark_result["access_time"].append(t1-t0)
            benchmark_result["data_processing"].append(t2-t1)
            benchmark_result["train_model"].append(t4-t3)
            benchmark_result["model_forecast"].append(t5-t4)
            benchmark_result["end_to_end"].append(t5-t0)
            
            logger.warning(benchmark_result)
            
        # Calculate mean times
        benchmark_results[cap] = {key: np.mean(value) for key, value in benchmark_result.items()}
        benchmark_results[cap]["end_to_end_std"] = np.std(benchmark_result["end_to_end"])
        logger.warning(benchmark_results)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(dir_path, output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Convert and write JSON object to file
    with open(os.path.join(out_dir,"benchmark_results_standard.json"), "w") as outfile: 
        json.dump(benchmark_results, outfile)
    plot_benchmark(benchmark_dict=benchmark_results,
                   out_dir=out_dir)
    logger.info("Benchmark completed. Results saved to {}", "benchmark_results.json")
        

        
        
    
    
   