import xarray as xr

class Bathymetry:
    def __init__(self, path: str):
        self.data = xr.open_dataset(path)

    def get_depth(self, lat: float, lon: float) -> float:
        """
        Returns the depth based on the provided latitude and longitude.

        :param lat: Latitude (in degrees).
        :param lon: Longitude (in degrees).
        :return: Corresponding depth (in meters).
        """

        # Get the latitude, longitude, and elevation variables
        latitudes = self.data['lat'].values
        longitudes = self.data['lon'].values
        elevations = self.data['elevation'].values

        # Find the nearest index for latitude and longitude
        lat_idx: float = np.abs(latitudes - lat).argmin()
        lon_idx: float = np.abs(longitudes - lon).argmin()

        # Return the depth (elevation) for the location
        depth: float = elevations[lat_idx, lon_idx]

        return depth

