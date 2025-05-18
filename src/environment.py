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

# Configuration parameters
min_freq = 10  # Hz
max_freq = 1000  # Hz
num_key_freq = 5  # Number of frequencies for Bellhop sampling
num_interp_freq = 100  # Total frequencies for final integration


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
        toa_variance: float,
        hydrophones=None,
        ships=None,
    ):
        self.area = area
        self.bathymetry = bathymetry
        self.noise_level = noise_level
        self.toa_variance = toa_variance

        self.ships = []
        self.hydrophones = []
        self.ac = AcousticCalculator()

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

    def calculate_pressures(self, include_dark=True):
        """
        Calculate pressures with frequency interpolation and proper TOA handling
        """
        # Create frequency grids
        key_frequencies = np.linspace(min_freq, max_freq, num_key_freq)
        interp_frequencies = np.linspace(min_freq, max_freq, num_interp_freq)

        # Convert noise to linear scale
        noise_std_linear = 10 ** (self.noise_level / 20) if self.noise_level > 0 else 0

        for hydro in self.hydrophones:
            total_observed_linear = 0.0
            toa_energy_pairs = []

            for ship in self.ships:
                if (not ship.is_dark) or include_dark:
                    pressure_values = []
                    toa_values = []

                    # 1. Get mock values at key frequencies
                    for f in key_frequencies:
                        pressure, toa = self.ac.calculate_linear_pressure(
                            f, self.bathymetry, ship.coord, hydro.coord
                        )
                        pressure_values.append(pressure)
                        toa_values.append(toa)

                    # 2. Interpolate pressure values across frequencies
                    interp_pressures = np.interp(
                        interp_frequencies, key_frequencies, pressure_values
                    )

                    # 3. Calculate energy (sum of squares)
                    ship_energy = np.sum([p**2 for p in interp_pressures])
                    ship_pressure = np.sqrt(ship_energy / num_interp_freq)

                    # 4. Find dominant TOA (use frequency with max pressure)
                    max_pressure_idx = np.argmax(pressure_values)
                    best_toa = toa_values[max_pressure_idx]
                    toa_energy_pairs.append((best_toa, ship_energy))

                    total_observed_linear += ship_pressure

            # 6. Determine overall best TOA
            if toa_energy_pairs:
                final_toa = min(toa_energy_pairs, key=lambda x: x[0])[0]
            else:
                final_toa = 0.0

            total_observed_linear += np.random.normal(0, noise_std_linear)
            final_toa += np.random.normal(0, np.sqrt(self.toa_variance))

            # 7. Store results
            result = {
                "toa": final_toa,
                "pressure": AcousticCalculator.linear_to_db(total_observed_linear),
            }
            hydro.observed_pressure.append(result)
