from arlpy.uwapm import check_env2d, create_env2d
from acoustic_calculator import AcousticCalculator
from bathymetry import Bathymetry
from hydrophone import Hydrophone
from point import Point
from ship import Ship

import numpy as np
from typing import Any, TypeAlias

# Enviroment area
GeoBoundingBox: TypeAlias = tuple[
    float, float, float, float
]  # lat_min, lat_max, lon_min, lon_max


class Environment:
    """Container for environment data"""

    area: GeoBoundingBox
    bathymetry: Bathymetry
    ships: list[Ship]
    hydrophones: list[Hydrophone]
    noise_level: float
    bellhop_env: Any

    def __init__(
        self, area: GeoBoundingBox, bathymetry: Bathymetry, noise_level: float
    ):
        self.area = area
        self.bathymetry = bathymetry
        self.noise_level = noise_level

        self.ships = []
        self.hydrophones = []
        self.bellhop_env = create_env2d()

    def get_random_coordinates(self):
        lat_rand = np.random.uniform(self.area[0], self.area[1])
        long_rand = np.random.uniform(self.area[2], self.area[3])
        return [lat_rand, long_rand]

    def set_area(self, area: GeoBoundingBox):
        self.area = area

    def set_noise_level(self, noise_level: float):
        self.noise_level = noise_level

    def set_bathymetry(self, bathymetry: Bathymetry):
        self.bathymetry = bathymetry

    def calculate_pressures(self):
        """
        Calculate expected and observed pressures for all hydrophones.
        :param self: environment
        """

        for hydro in self.hydrophones:
            total_observed_linear = 0.0
            total_expected_linear = 0.0

            for ship in self.ships:
                env = self.get_bellhop_env(ship.coord, hydro.coord)

                # Calculate linear pressure received from the ship
                received_pressure = AcousticCalculator.calculate_linear_pressure(
                    hydro, ship, env
                )

                # Sum the linear pressures
                total_observed_linear += received_pressure
                if not ship.is_dark:
                    total_expected_linear += received_pressure


            # Convert total observed pressure to dB re 1 µPa
            hydro.observed_pressure = AcousticCalculator.linear_to_db(
                total_observed_linear
            )

            hydro.observed_pressure += np.random.normal(0, self.noise_level)

            # Convert total expected pressure to dB re 1 µPa
            hydro.expected_pressure = AcousticCalculator.linear_to_db(
                total_expected_linear
            )

    def add_ship(self, ship: Ship):
        self.ships.append(ship)

    def add_hydrophone(self, hydrophone: Hydrophone):
        self.hydrophones.append(hydrophone)

    def get_bellhop_env(self, ship_coord: Point, hydro_coord: Point):
        env = create_env2d()
        env["depth"] = self.bathymetry.get_depth_profile(ship_coord, hydro_coord, 10)
        env["tx_depth"] = ship_coord.depth
        env["rx_depth"] = hydro_coord.depth
        #env["rx_range"] = ship_coord.distance(hydro_coord)

        # check_env2d(env)

        return env
