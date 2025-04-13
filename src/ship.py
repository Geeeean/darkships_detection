from point import Point


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
