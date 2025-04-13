from math import log10, sqrt
import arlpy.uwapm as pm
import numpy as np


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
    def calculate_attenuation(env):
        """
        Calculate the attenuation of acoustic pressure due to geometric spreading.
        :param distance: Distance between the source and the receiver in meters.
        :return: Attenuation in dB.
        """
        tl_complex = pm.compute_transmission_loss(env=env, mode="incoherent").values[0][0]
        return -20 * np.log10(np.abs(tl_complex) + 1e-12)

    @staticmethod
    def calculate_linear_pressure(
        frequency: float, ship_density: float, bandwith: float, env
    ):
        """
        Calculate the linear pressure received by a hydrophone from a ship.
        :param frequency: Central frequency in Hz
        :param ship_density: Ship density in ships/km²
        :param bandwidth: Frequency bandwidth in Hz
        :param env: Bellhop environment object
        :return: Linear pressure in µPa
        """

        # calculate attenuation of the pressure based on the distance
        attenuation = AcousticCalculator.calculate_attenuation(env)  # [dB]
        attenuation_linear = AcousticCalculator.db_to_linear(
            attenuation
        )  # [adimensional]

        ship_pressure = AcousticCalculator.shipping_noise_ross(
            frequency, ship_density
        )  # [dB re µPa/√Hz]
        ship_linear_pressure = AcousticCalculator.db_to_linear(
            ship_pressure
        )  # [µPa/√Hz]
        tot_pressure = ship_linear_pressure * sqrt(bandwith)  # [µPa]

        return tot_pressure / attenuation_linear  # [µPa]

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
