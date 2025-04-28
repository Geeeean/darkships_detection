from math import floor, log10, sqrt
import arlpy.uwapm as pm
import numpy as np

from bathymetry import Bathymetry
from hydrophone import Hydrophone
from point import Point


class AcousticCalculator:
    def __init__(self) -> None:
        self.attenuation_cache = dict()

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
    def calculate_attenuation(env):
        """
        Calculate the attenuation of acoustic pressure due to geometric spreading.
        :param distance: Distance between the source and the receiver in meters.
        :return: Attenuation in dB.
        """
        tl_complex = pm.compute_transmission_loss(env=env, mode="incoherent").values[0][
            0
        ]
        return -20 * np.log10(np.abs(tl_complex) + 1e-12)

    def calculate_linear_pressure(
        self,
        frequency: float,
        ship_density: float,
        bandwith: float,
        bathymetry: Bathymetry,
        ship_point: Point,
        hydro_point: Point,
    ):
        """
        Calculate the linear pressure received by a hydrophone from a ship.
        :param frequency: Central frequency in Hz
        :param ship_density: Ship density in ships/km²
        :param bandwidth: Frequency bandwidth in Hz
        :param env: Bellhop environment object
        :return: Linear pressure in µPa
        """

        ship_pressure = AcousticCalculator.shipping_noise_ross(
            frequency, ship_density
        )  # [dB re µPa/√Hz]
        ship_pressure = 100
        ship_linear_pressure = AcousticCalculator.db_to_linear(
            ship_pressure
        )  # [µPa/√Hz]
        tot_pressure = ship_linear_pressure * sqrt(bandwith)  # [µPa]

        # Estimating attenuation before running bellhop
        # bellhop requires high computational cost: high attenuation estim
        # allows to skip this passage
        # DELTA_TRESHOLD = 10 # [µPa]
        # alpha = AcousticCalculator.absorption_coefficient_thorp(frequency)
        # distance_km = env["rx_range"] / 1000  # [km]
        # estimated_tl = 20 * log10(distance_km * 1000) + alpha * distance_km
        # estimated_attenuation = AcousticCalculator.db_to_linear(estimated_tl)
        # estimated_received_pressure = tot_pressure / estimated_attenuation

        # Skip Bellhop: signal too weak to be relevant
        # if estimated_received_pressure < DELTA_TRESHOLD:
        #     return 0.0

        attenuation_linear = self.attenuation_cache.get((ship_point, hydro_point))
        if attenuation_linear is None:
            env = AcousticCalculator.get_bellhop_env(
                bathymetry, ship_point, hydro_point, frequency
            )

            # calculate attenuation of the pressure based on the distance
            attenuation = AcousticCalculator.calculate_attenuation(env)  # [dB]
            attenuation_linear = AcousticCalculator.db_to_linear(
                attenuation
            )  # [adimensional]

            self.attenuation_cache[(ship_point, hydro_point)] = attenuation_linear

        pressure = tot_pressure / attenuation_linear  # [µPa]

        return pressure

    @staticmethod
    def shipping_noise_ross(frequency, ship_density):
        """
        Compute ship noise using Ross formula

        Args:
            frequency (float): [Hz]
            ship_density (float): [ship/km²]

        Returns:
            nl: Ship noise in dB re µPa/√Hz.
        """
        f_kHz = frequency / 1000  # Hz -> KHz
        nl = 40 + 20 * np.log10(f_kHz) - 17 * np.log10(ship_density + 1e-12)
        return nl

    @staticmethod
    def absorption_coefficient_thorp(frequency_hz: float) -> float:
        """
        Compute frequency-dependent absorption coefficient (Thorp formula).
        :param frequency_hz: Frequency in Hz
        :return: Attenuation in dB/km
        """
        f = frequency_hz / 1000  # convert to kHz

        alpha = (
            (0.11 * f**2) / (1 + f**2)
            + (44 * f**2) / (4100 + f**2)
            + 0.000275 * f**2
            + 0.003
        )
        return alpha  # [dB/km]

    @staticmethod
    def get_bellhop_env(
        bathymetry: Bathymetry, ship_coord: Point, hydro_coord: Point, frequency: float
    ):
        env = pm.create_env2d()

        bathy = bathymetry.get_depth_profile(ship_coord, hydro_coord, 10)
        env["depth"] = Bathymetry.bellhop_sanitized(bathy)

        env["tx_depth"] = ship_coord.depth
        env["rx_depth"] = hydro_coord.depth
        env["frequency"] = frequency
        env["rx_range"] = floor(ship_coord.distance_2d(hydro_coord))

        pm.check_env2d(env)

        return env
