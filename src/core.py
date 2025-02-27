import numpy as np
from math import log10

# NOISE_TRESHOLD = 5 # [db]

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
    def calculate_noises(hydrophones, ships, config):
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
            hydro.observed_noise += np.random.normal(0, noise_level)

            hydro.expected_noise = 10 * np.log10(total_expected_linear + 1e-12)

    @staticmethod
    def compute_noies_delta(hydro):
        """Calculate difference between observed and expected noise"""
        return hydro.observed_noise - hydro.expected_noise

