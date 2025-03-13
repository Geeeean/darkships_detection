import numpy as np
from scipy.optimize import minimize

from acoustic_calculator import AcousticCalculator
from utils import Position

class DarkShipTracker:
    @staticmethod
    def mlat(hydrophones):
        """
        Estimate the position of the ship using triangulation based on hydrophone pressure differences.

        :param hydrophones: List of hydrophone objects with observed and expected acoustic pressure properties.
        :return: Estimated (x, y) position of the ship and estimated base pressure.
        """

        def loss_function(params):
            """Calculate error between estimated and observed pressure deltas."""
            ship_x, ship_y, ship_pressure = params
            ship_pos = Position(ship_x, ship_y)

            total_error = 0

            for hydro in hydrophones:
                # Calculate distance from ship to hydrophone
                distance = Position.distance(ship_pos, hydro)

                # Compute expected pressure delta using the inverse model
                darkship_observed_pressure = ship_pressure - AcousticCalculator.calculate_attenuation(distance)

                # Computing the observed value as the sum of the expected value and the darkship observed pressure
                darkship_observed_linear = AcousticCalculator.db_to_linear(darkship_observed_pressure)
                hydro_expected_linear = AcousticCalculator.db_to_linear(hydro.expected_pressure)

                new_observation = AcousticCalculator.linear_to_db(darkship_observed_linear + hydro_expected_linear)

                # Compute squared error
                total_error += (new_observation - hydro.observed_pressure) ** 2

            return total_error

        # Initial guess (centered in the middle of hydrophones)
        x0 = np.mean([h.x for h in hydrophones])
        y0 = np.mean([h.y for h in hydrophones])

        DEFAULT_PRESSURE = 150

        # Optimize (x, y) position and base acoustic pressure of the ship
        result = minimize(loss_function, [x0, y0, DEFAULT_PRESSURE], method='Nelder-Mead')

        # Return estimated ship position and its estimated base pressure
        return result.x[:2], result.x[2]
