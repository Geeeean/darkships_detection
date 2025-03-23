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

    @property
    def latitude(self):
        return self.coord.latitude

    @property
    def longitude(self):
        return self.coord.longitude

    def get_depth(self):
        return self.depth
