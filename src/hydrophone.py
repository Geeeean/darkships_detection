from point import Point


class Hydrophone:
    """Represents an underwater acoustic sensor
    Attributes:
        id (int): Unique identifier
        coord (Point): Lat, Long and Point
        observed_pressure (float): Measured acoustic pressure [dB re 1μPa]
        expected_pressure (float): Predicted acoustic pressure from AIS data
        max_range (float): Max range the hydrophone can measure pressure [km]
    """

    def __init__(
        self, id: int, lat: float, long: float, depth: float, max_range: float
    ):
        self.id = id
        self.coord = Point(lat, long, depth)
        self.max_range = max_range
        self.observed_pressure = 0.0
        self.expected_pressure = 0.0
