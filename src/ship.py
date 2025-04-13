from point import Point
from math import cos, radians, sin


class Ship:
    """Represents a vessel with acoustic properties
    Attributes:
        id (int): Unique identifier
        coord (Point): Lat, Long and Depth
        speed (float): Speed [knots]
        is_dark (bool): True if not transmitting AIS
        heading (float): Compass direction in which the craft's bow or nose is pointed [°]
    """

    def __init__(
        self,
        id: int,
        lat: float,
        long: float,
        depth: float,
        speed: float,
        heading: float,
        is_dark: bool = False,
    ):
        self.id = id
        self.coord = Point(lat, long, depth)
        self.speed = speed
        self.is_dark = is_dark
        self.heading = heading

    def update_position(self, delta_t_sec: float):
        """
        Update the ship position based on his speed and heading
        :param delta_t_sec: time interval [s]
        """
        # [kt] -> [m/s]
        speed_mps = self.speed * 0.514444

        distance_m = speed_mps * delta_t_sec

        lat_change = distance_m * cos(radians(self.heading)) / 111320
        lon_change = (
            distance_m
            * sin(radians(self.heading))
            / (111320 * cos(radians(self.coord.latitude)))
        )

        self.coord.latitude += lat_change
        self.coord.longitude += lon_change
