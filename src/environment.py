from math import floor, pi, sqrt
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
    ac: AcousticCalculator

    def __init__(
        self,
        area: GeoBoundingBox,
        bathymetry: Bathymetry,
        noise_level: float,
        hydrophones=None,
        ships=None,
    ):
        self.area = area
        self.bathymetry = bathymetry
        self.noise_level = noise_level

        self.ships = []
        self.hydrophones = []
        self.ac = AcousticCalculator()

        # constants
        self.bandwith = 100  # [Hz]
        self.frequencies = [100, 500, 1000]  # [Hz]

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

    def calculate_pressures(self, include_dark=True):
        """
        Calculate expected and observed pressures for all hydrophones.
        :param self: environment
        """

        for hydro in self.hydrophones:
            total_observed_linear = 0.0
            total_expected_linear = 0.0

            ship_density = self.calculate_ship_density(hydro)

            for ship in self.ships:
                if (not ship.is_dark) or include_dark:
                    # Calculate linear pressure received from the ship
                    p_tot = 0
                    for frequency in self.frequencies:
                        (pressure, toa) = self.ac.calculate_linear_pressure(
                            frequency,
                            ship_density,
                            self.bandwith,
                            self.bathymetry,
                            ship.coord,
                            hydro.coord,
                        )

                        p_tot += pressure**2

                    p_tot = sqrt(p_tot)

                    # Sum the linear pressures
                    total_observed_linear += p_tot
                    if not ship.is_dark:
                        total_expected_linear += p_tot

            # Convert total observed pressure to dB re 1 µPa
            hydro.observed_pressure.append(
                {
                    "toa": toa,
                    "pressure": AcousticCalculator.linear_to_db(total_observed_linear)
                    + np.random.normal(0, self.noise_level),
                }
            )

            # Convert total expected pressure to dB re 1 µPa
            # hydro.expected_pressure = AcousticCalculator.linear_to_db(
            #     total_expected_linear
            # )

    def add_ship(self, ship: Ship):
        self.ships.append(ship)

    def add_hydrophone(self, hydrophone: Hydrophone):
        self.hydrophones.append(hydrophone)

    def calculate_ship_density(self, hydro: Hydrophone):
        counter = 0
        radius = 50  # [km]

        for ship in self.ships:
            if (
                ship.coord.distance_2d(hydro.coord) < radius * 1000
            ):  # radius [km] -> [m]
                counter += 1

        return counter / (pi * radius**2)
