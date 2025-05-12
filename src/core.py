import numpy as np

from environment import Environment


class DarkShipTracker:
    @staticmethod
    def weighted_centroid_localization(environment: Environment):
        w_lat = 0
        w_lon = 0
        delta_tot = 0
        for hydro in environment.hydrophones:
            delta = hydro.compute_pressure_delta()
            delta_tot += delta

            w_lat_i = hydro.coord.latitude * delta
            w_lon_i = hydro.coord.longitude * delta

            w_lat += w_lat_i
            w_lon += w_lon_i

        if delta_tot == 0:
            return (0, 0)

        w_avg_lat = w_lat / delta_tot
        w_avg_lon = w_lon / delta_tot

        return (w_avg_lat, w_avg_lon)
