from point import Point

class Ship:
    """Represents a vessel with acoustic properties
    Attributes:
        id (int): Unique identifier
        coord (Point): Lat, Long and Depth
        speed (float): Speed [knots]
        is_dark (bool): True if not transmitting AIS
        base_pressure (float): Acoustic pressure at 1m distance
    """

    def __init__(
        self,
        id: int,
        lat: float,
        long: float,
        depth: float,
        speed: float,
        is_dark: bool = False,
        base_pressure: float = 140,
    ):
        self.id = id
        self.coord = Point(lat, long, depth)
        self.speed = speed
        self.is_dark = is_dark
        self.base_pressure = (
            base_pressure + 0.5 * speed
        )  # Empirical pressure-speed relationship
