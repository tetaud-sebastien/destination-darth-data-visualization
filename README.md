<table style="width:100%; border: none;">
    <tr>
        <td colspan="3" style="text-align:center; border: none;">
            <img src="assets/banner.svg" alt="Banner Image" style="width:100vh;" >
        </td>
    </tr>
    <!-- Add other rows and cells below if needed -->
</table>

# DestinE Climate Data Visualization

## Overview

This repository provides the necessary tools to benchmark DestinE data access services, including polytope and cacheB, and visualize the data on an interactive map using Folium. The project aims to facilitate the analysis and comparison of different data access methods, as well as the presentation of climate data in a user-friendly manner.

## Features

- **Benchmarking Tools:** Perform benchmarks between different DestinE data access services. CacheB and Polytope only supported at the moment.
- **Data Visualization:** Visualize climate data on an interactive map with markers for different capitals.
- **Forecast Plotting:** Generate and display time series forecast for specific locations.

## Prerequisites
1. Clone the repository:
    ```bash
    git clone git@github.com:tetaud-sebastien/Destination-Earth-Climate-Data-Visualization.git
    ```
2. Install Python
    Download and install Python
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh
    ```
3. Install the required packages
    Create python environment:
    ```bash
    conda create --name destine_env python==3.11
    ```
    Activate the environment

    ```bash
    conda activate destine_env
    ```
    Install python package
    ```Bash
    pip install -r requirements.txt
    ```

## Service authentification

```Bash
python authentification/cacheb-authentication.py -u username -p password >> ~/.netrc
python authentification/desp-authentication.py --user username --password password
```

## Usage

### Benchmarking Data Access Services

```Bash
python main.py
```
**main.py** takes as input a yaml file as input:

```yaml
request_nb: 2
cacheb_url: https://cacheb.dcms.e2e.desp.space/destine-climate-dt/SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr
capital_coordinates: {
    "Vienna": [48.2082, 16.3738],
    "Brussels": [50.8503, 4.3517],
}
output_folder: "result"
```

where:

- **request_nb**: Number of request for the same product to perform statistics
- **cacheb_url**: cacheB url to access the dataset
- **capital_coordinates**: dictionary type that contain the name of cities and its associated lat lon
- **output_folder**: folder name that will be created in the root project foler.

