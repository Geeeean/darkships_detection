from math import sqrt
from geopy.point import Point as Geopoint
from geopy.distance import geodesic


class Point:
    """Represents a point in the sea
    Attributes:
        coord (Point): Lat, Long
        depth (float): depth under the sea surface [m]
    """

    def __init__(self, lat: float, long: float, depth: float):
        self.coord: Geopoint = Geopoint(lat, long)
        self.depth: float = depth

    def distance(self, other: "Point") -> float:
        d2d = geodesic(self.coord, other.coord).meters
        delta_depth: float = abs(self.depth - other.depth)
        return sqrt(d2d**2 + delta_depth**2)

    def distance_2d(self, other: "Point") -> float:
        return geodesic(self.coord, other.coord).meters

    @property
    def latitude(self):
        """Getter for latitude."""
        return self.coord.latitude

    @property
    def longitude(self):
        """Getter for longitude."""
        return self.coord.longitude

    @latitude.setter
    def latitude(self, value: float):
        """Setter for latitude."""
        self.coord = Geopoint(value, self.coord.longitude)

    @longitude.setter
    def longitude(self, value: float):
        """Setter for longitude."""
        self.coord = Geopoint(self.coord.latitude, value)
