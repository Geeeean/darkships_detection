from math import sqrt
from arlpy.uwapm import print_env
import numpy as np
from acoustic_calculator import AcousticCalculator
from point import Point
from environment import Environment
from scipy.optimize import minimize


class DarkShipTracker:
    @staticmethod
    def mlat(environment: Environment):
        """
        Estimate the position of the ship using triangulation based on hydrophone pressure differences.

        :param hydrophones: List of hydrophone objects with observed and expected acoustic pressure properties.
        :return: Estimated (x, y) position of the ship and estimated base pressure.
        """

        def loss_function(params: list[float]):
            """Calculate error between estimated and observed pressure deltas."""

            ship_lat, ship_long, ship_depth = params
            ship_point = Point(ship_lat, ship_long, ship_depth)

            total_error = 0

            for hydro in environment.hydrophones:
                ship_density = environment.calculate_ship_density(hydro)

                p_tot = 0
                for frequency in environment.frequencies:
                    env = environment.get_bellhop_env(
                        ship_point, hydro.coord, frequency
                    )
                    p_tot += (
                        AcousticCalculator.calculate_linear_pressure(
                            frequency, ship_density, environment.bandwith, env
                        )
                        ** 2
                    )

                darkship_observed_linear = sqrt(p_tot)

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

        # Optimize (x, y) position and base acoustic pressure of the ship
        result = minimize(loss_function, [lat, long, depth], method="Nelder-Mead")

        # Return estimated ship position and its estimated base pressure
        return result.x[:3], result.x[2]
