from arlpy import uwapm as pm
import numpy as np
from scipy.optimize import minimize

from acoustic_calculator import AcousticCalculator
from point import Point
from environment import Environment


class DarkShipTracker:
    @staticmethod
    def mlat(environment: Environment):
        """
        Estimate the position of the ship using triangulation based on hydrophone pressure differences.

        :param hydrophones: List of hydrophone objects with observed and expected acoustic pressure properties.
        :return: Estimated (x, y) position of the ship and estimated base pressure.
        """

        bellhop_env = pm.create_env2d()

        def loss_function(params: list[float]):
            """Calculate error between estimated and observed pressure deltas."""
            ship_lat, ship_long, ship_depth, ship_pressure = params
            ship_point = Point(ship_lat, ship_long, ship_depth)

            total_error = 0

            for hydro in environment.hydrophones:
                bellhop_env["depth"] = environment.bathymetry.get_depth_profile(
                    ship_point, hydro.coord, 10
                )
                bellhop_env["tx_depth"] = ship_point.depth
                bellhop_env["rx_depth"] = hydro.coord.depth

                attenuation = AcousticCalculator.calculate_attenuation(bellhop_env)

                darkship_observed_pressure = ship_pressure - attenuation

                # Computing the observed value as the sum of the expected value and the darkship observed pressure
                darkship_observed_linear = AcousticCalculator.db_to_linear(
                    darkship_observed_pressure
                )
                hydro_expected_linear = AcousticCalculator.db_to_linear(
                    hydro.expected_pressure
                )

                new_observation = AcousticCalculator.linear_to_db(
                    darkship_observed_linear + hydro_expected_linear
                )

                # Compute squared error
                total_error += (new_observation - hydro.observed_pressure) ** 2

            return total_error

        # Initial guess (centered in the middle of hydrophones)
        lat = np.mean([h.coord.latitude for h in environment.hydrophones])
        long = np.mean([h.coord.longitude for h in environment.hydrophones])
        depth = np.mean([h.coord.depth for h in environment.hydrophones])

        DEFAULT_PRESSURE = 150

        # Optimize (x, y) position and base acoustic pressure of the ship
        result = minimize(
            loss_function, [lat, long, depth, DEFAULT_PRESSURE], method="Nelder-Mead"
        )

        # Return estimated ship position and its estimated base pressure
        return result.x[:3], result.x[2]
