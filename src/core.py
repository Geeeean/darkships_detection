import numpy as np
from scipy.optimize import minimize

from acoustic_calculator import AcousticCalculator

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
            total_error = 0

            for hydro in hydrophones:
                # Get hydrophone position
                hx, hy = hydro.x, hydro.y

                # Compute pressure delta using the static method
                pressure_delta = AcousticCalculator.compute_pressure_delta(hydro)

                # Calculate distance from ship to hydrophone
                distance = np.sqrt((ship_x - hx)**2 + (ship_y - hy)**2)

                # Compute expected pressure delta using the inverse model
                estimated_pressure = ship_pressure - 20 * np.log10(distance + 1e-9)

                # Compute squared error
                total_error += (estimated_pressure - pressure_delta) ** 2

            return total_error

        # Initial guess (centered in the middle of hydrophones)
        x0 = np.mean([h.x for h in hydrophones])
        y0 = np.mean([h.y for h in hydrophones])

        DEFAULT_PRESSURE = 150

        # Optimize (x, y) position and base acoustic pressure of the ship
        result = minimize(loss_function, [x0, y0, DEFAULT_PRESSURE], method='Nelder-Mead')

        # Return estimated ship position and its estimated base pressure
        return result.x[:2], result.x[2]
