from math import log10
import arlpy.uwapm as pm

from hydrophone import Hydrophone
from ship import Ship


class AcousticCalculator:
    @staticmethod
    def db_to_linear(pressure_db: float):
        """
        Convert pressure from dB re 1 µPa to linear scale (µPa).
        :param pressure_db: Pressure in dB re 1 µPa.
        :return: Pressure in µPa.
        """
        return 10 ** (pressure_db / 20)

    @staticmethod
    def linear_to_db(pressure_linear: float):
        """
        Convert pressure from linear scale (µPa) to dB re 1 µPa.
        :param pressure_linear: Pressure in µPa.
        :return: Pressure in dB re 1 µPa.
        """
        return 20 * log10(pressure_linear + 1e-12)  # Add 1e-12 to avoid log(0)

    @staticmethod
    def calculate_attenuation(distance: float):
        """
        Calculate the attenuation of acoustic pressure due to geometric spreading.
        :param distance: Distance between the source and the receiver in meters.
        :return: Attenuation in dB.
        """
        return 20 * log10(distance + 1e-9)  # Avoid log(0)

    @staticmethod
    def calculate_linear_pressure(hydro: Hydrophone, ship: Ship, env):
        """
        Calculate the linear pressure received by a hydrophone from a ship.
        :param hydro: Hydrophone object with x, y coordinates.
        :param ship: Ship object with x, y coordinates and base pressure level.
        :return: Linear pressure in µPa.
        """

        # Distance between the hydrophone and the ship
        distance = hydro.coord.distance(ship.coord)

        # calculate attenuation of the pressure based on the distance
        attenuation = AcousticCalculator.calculate_attenuation(distance)

        received_pressure_db = ship.base_pressure - attenuation

        # Convert received pressure from dB to linear scale (µPa)
        received_pressure_linear = AcousticCalculator.db_to_linear(received_pressure_db)

        return received_pressure_linear

    @staticmethod
    def calculate_distance_from_pressure(
        hydro_pressure: float, ship_base_pressure: float
    ):
        """
        Calculate the distance from a hydrophone to a ship based on the received pressure.

        :param hydro_pressure: Observed acoustic pressure at the hydrophone (dB).
        :param ship_base_pressure: Base acoustic pressure of the ship (dB).
        :return: Estimated distance (in the same unit as used in original calculations).
        """

        # Compute the distance
        log_distance = (ship_base_pressure - hydro_pressure) / 20
        distance = 10**log_distance - 1e-9

        return max(distance, 0)  # Ensure distance is non-negative
