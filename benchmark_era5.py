import cdsapi
from loguru import logger
import xarray as xr


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
        self.result = self.client.retrieve(name, request)
        return self.result

    def download(self, filename):
        """
        """
        self.filename = filename
        self.result.download(filename)

    def process(self,lat: float = 48.8566, lon: float = 2.3522,
               method: str = "nearest"):


        ds = xr.open_dataset(self.filename, engine="cfgrib")
        ds = ds.sel(latitude=lat, longitude=lon, method=method)
        return ds


if __name__ == "__main__":


    query = {"name":"reanalysis-era5-pressure-levels",
             "request":{
             "variable": "temperature",
                "pressure_level": "1000",
                "product_type": "reanalysis",
                "year": "2010",
                "month": "01",
                "day": "01",
                "time": "12:00",
                "format": "grib"
                }}

    cds = CdsERA5()
    cds.get_data(query=query)
    cds.download(filename="download.grib")
    ds = cds.process()

















