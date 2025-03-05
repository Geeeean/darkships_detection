import numpy as np
from math import log10
from scipy.optimize import minimize

class AcousticCalculator:
    @staticmethod
    def _calculate_linear_noise(hydro, ship):
        # calculate the distance between the hydrophone and the ship
        dx = hydro.x - ship.x
        dy = hydro.y - ship.y
        distance = (dx**2 + dy**2)**0.5

        # calculate attenuation of the noise based on the distance
        attenuation = 20 * log10(distance + 1e-9) # sum 1e-9 to avoid log(0)

        # decibels are logarithmic => transform in the linear form
        received_power = 10 ** ((ship.base_noise - attenuation) / 10)

        return received_power

    @staticmethod
    def calculate_noises(hydrophones, ships, config, include_base_noise_level = False):
        """Calculate expected and observed noise for all hydrophones"""
        noise_level = config['hydrophones'].get('noise_level', 0.0)

        for hydro in hydrophones:
            total_observed_linear = 0.0
            total_expected_linear = 0.0

            for ship in ships:
                received_power = AcousticCalculator._calculate_linear_noise(hydro, ship)

                total_observed_linear += received_power
                if not ship.is_dark:
                    total_expected_linear += received_power

            # Convert from linear form to db and add the noise level
            hydro.observed_noise = 10 * np.log10(total_observed_linear + 1e-12)
            if include_base_noise_level:
                hydro.observed_noise += np.random.normal(0, noise_level)

            hydro.expected_noise = 10 * np.log10(total_expected_linear + 1e-12)

    @staticmethod
    def calculate_distance_from_noise(hydro_noise, ship_noise_base):
        """
        Calculate the distance from a hydrophone to a ship based on the received noise level.

        :param hydro_noise: Observed noise level at the hydrophone (in dB).
        :param ship_noise_base: Base noise level of the ship (in dB).
        :return: Estimated distance (in the same unit as used in original calculations).
        """
        # Convert noise to linear scale
        received_power = 10 ** (hydro_noise / 10)

        # Compute the distance using the correct formula
        log_distance = (ship_noise_base - 10 * np.log10(received_power)) / 20
        distance = 10 ** log_distance - 1e-9

        return max(distance, 0)  # Ensure distance is non-negative

    @staticmethod
    def compute_noise_delta(hydro):
        """Calculate difference between observed and expected noise"""
        return hydro.observed_noise - hydro.expected_noise

class DarkShipTracker:
    @staticmethod
    def mlat(hydrophones):
        """
        Estimate the position of the ship using triangulation based on hydrophone noise differences.

        :param hydrophones: List of hydrophone objects with observed and expected noise properties.
        :return: Estimated (x, y) position of the ship and estimated noise base.
        """

        def loss_function(params):
            """Calculate error between estimated and observed noise deltas."""
            ship_x, ship_y, ship_noise = params
            total_error = 0

            for hydro in hydrophones:
                # Get hydrophone position
                hx, hy = hydro.x, hydro.y

                # Compute noise delta using the static method
                noise_delta = AcousticCalculator.compute_noise_delta(hydro)

                # Calculate distance from ship to hydrophone
                distance = np.sqrt((ship_x - hx)**2 + (ship_y - hy)**2)

                # Compute expected noise delta using the inverse model
                estimated_noise = ship_noise - 20 * np.log10(distance + 1e-9)

                # Compute squared error
                total_error += (estimated_noise - noise_delta) ** 2

            return total_error

        # Initial guess (centered in the middle of hydrophones)
        x0 = np.mean([h.x for h in hydrophones])
        y0 = np.mean([h.y for h in hydrophones])

        DEFAULT_NOISE = 150

        # Optimize (x, y) position and noise base level
        result = minimize(loss_function, [x0, y0, DEFAULT_NOISE], method='Nelder-Mead')

        # Return estimated ship position and noise base
        return result.x[:2], result.x[2]  # (x, y), estimated noise base
