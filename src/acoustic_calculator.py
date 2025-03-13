import numpy as np
from math import log10, sqrt

class AcousticCalculator:
    @staticmethod
    def db_to_linear(pressure_db):
        """
        Convert pressure from dB re 1 µPa to linear scale (µPa).
        :param pressure_db: Pressure in dB re 1 µPa.
        :return: Pressure in µPa.
        """
        return 10 ** (pressure_db / 20)

    @staticmethod
    def linear_to_db(pressure_linear):
        """
        Convert pressure from linear scale (µPa) to dB re 1 µPa.
        :param pressure_linear: Pressure in µPa.
        :return: Pressure in dB re 1 µPa.
        """
        return 20 * log10(pressure_linear + 1e-12)  # Add 1e-12 to avoid log(0)

    @staticmethod
    def calculate_attenuation(distance):
        """
        Calculate the attenuation of acoustic pressure due to geometric spreading.
        :param distance: Distance between the source and the receiver in meters.
        :return: Attenuation in dB.
        """
        return 20 * log10(distance + 1e-9) # Avoid log(0)

    @staticmethod
    def calculate_linear_pressure(hydro, ship):
        """
        Calculate the linear pressure received by a hydrophone from a ship.
        :param hydro: Hydrophone object with x, y coordinates.
        :param ship: Ship object with x, y coordinates and base pressure level.
        :return: Linear pressure in µPa.
        """

        # Distance between the hydrophone and the ship
        dx = hydro.x - ship.x
        dy = hydro.y - ship.y
        distance = sqrt(dx**2 + dy**2)

        # calculate attenuation of the pressure based on the distance
        attenuation = AcousticCalculator.calculate_attenuation(distance)

        received_pressure_db = ship.base_pressure - attenuation

        # Convert received pressure from dB to linear scale (µPa)
        received_pressure_linear = AcousticCalculator.db_to_linear(received_pressure_db)

        return received_pressure_linear

    @staticmethod
    def calculate_pressures(hydrophones, ships, config, include_base_noise_level = False):
        """
        Calculate expected and observed pressures for all hydrophones.
        :param hydrophones: List of hydrophone objects.
        :param ships: List of ship objects.
        :param config: Configuration dictionary.
        :param include_base_noise_level: Whether to include random noise in the observed pressure.
        """
        noise_level = config['hydrophones'].get('noise_level', 0.0)

        for hydro in hydrophones:
            total_observed_linear = 0.0
            total_expected_linear = 0.0

            for ship in ships:
                # Calculate linear pressure received from the ship
                received_pressure = AcousticCalculator.calculate_linear_pressure(hydro, ship)

                # Sum the linear pressures
                total_observed_linear += received_pressure
                if not ship.is_dark:
                    total_expected_linear += received_pressure

            # Convert total observed pressure to dB re 1 µPa
            hydro.observed_pressure = AcousticCalculator.linear_to_db(total_observed_linear)

            if include_base_noise_level:
                # Add random noise to the observed pressure
                hydro.observed_pressure += np.random.normal(0, noise_level)

            # Convert total expected pressure to dB re 1 µPa
            hydro.expected_pressure = AcousticCalculator.linear_to_db(total_expected_linear)

    @staticmethod
    def calculate_distance_from_pressure(hydro_pressure, ship_base_pressure):
        """
        Calculate the distance from a hydrophone to a ship based on the received pressure.

        :param hydro_pressure: Observed acoustic pressure at the hydrophone (dB).
        :param ship_base_pressure: Base acoustic pressure of the ship (dB).
        :return: Estimated distance (in the same unit as used in original calculations).
        """

        # Compute the distance
        log_distance = (ship_base_pressure - hydro_pressure) / 20
        distance = 10 ** log_distance - 1e-9

        return max(distance, 0)  # Ensure distance is non-negative

    @staticmethod
    def compute_pressure_delta(hydro):
        """Calculate difference between observed and expected acoustic pressure"""
        return hydro.observed_pressure - hydro.expected_pressure

