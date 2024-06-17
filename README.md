<table style="width:100%; border: none;">
    <tr>
        <td colspan="3" style="text-align:center; border: none;">
            <img src="assets/banner.svg" alt="Banner Image" style="width:100vh;" >
        </td>
    </tr>
    <!-- Add other rows and cells below if needed -->
</table>

# DestinE Data Visualization

## Prerequisites
1. Clone the repository:
    ```bash
    git clone git@github.com:tetaud-sebastien/destination-earth-climate-data-visualization.git
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

