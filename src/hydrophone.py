from point import Point


class Hydrophone:
    """Represents an underwater acoustic sensor
    Attributes:
        id (int): Unique identifier
        coord (Point): Lat, Long and Point
        observed_pressure ([float]): Measured acoustic pressure during time [dB re 1μPa]
        expected_pressure (float): Predicted acoustic pressure from AIS data
        max_range (float): Max range the hydrophone can measure pressure [km]
    """

    def __init__(self, id: int, lat: float, long: float, depth: float):
        self.id = id
        self.coord = Point(lat, long, depth)
        self.observed_pressure = []
        # self.expected_pressure = 0.0

    def compute_pressure_delta(self):
        """Calculate difference between last and first observed pressure"""
        if not self.observed_pressure:
            return 0

        first = self.observed_pressure[0]["pressure"]
        last = self.observed_pressure[-1]["pressure"]

        return last - first
