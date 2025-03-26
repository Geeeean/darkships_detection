from typing import List
import xarray as xr
import numpy as np

from point import Point

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

    def get_depth_profile(self, start_coords: Point, end_coords: Point, num_points: int = 10) -> List[List[float]]:
        """
        Generates a depth profile between two points with distances from start point.

        Args:
            start_coords: Point - Ship coordinates
            end_coords: Point - Hydrophone coordinates
            num_points: Number of sampling points along the path

        Returns:
            List of [distance_from_start_m, depth] pairs in meters
            Example:
            [
                [0, 30],    # 30 m at start point (ship)
                [300, 20],  # 20 m 300m from ship
                [1000, 25]  # 25 m at 1km
            ]
        """
        # Calculate intermediate points along the great circle path
        lats = np.linspace(start_coords.latitude, end_coords.longitude, num_points)
        lons = np.linspace(start_coords.latitude, end_coords.longitude, num_points)

        # Create temporary points for each intermediate location
        intermediate_points = [
            Point(lat, lon, self.get_depth(lat, lon))
            for lat, lon in zip(lats, lons)
        ]

        # Calculate cumulative distances
        profile = []
        cumulative_distance = 0.0

        # Add starting point
        profile.append([0.0, start_coords.depth])

        # Calculate distances between consecutive points
        for i in range(1, len(intermediate_points)):
            segment_distance = intermediate_points[i-1].distance(intermediate_points[i])
            cumulative_distance += segment_distance
            profile.append([cumulative_distance, intermediate_points[i].depth])

        return profile
