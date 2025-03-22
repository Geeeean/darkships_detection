import numpy as np
import matplotlib.pyplot as plt
import yaml

import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

from utils import Utils
from acoustic_calculator import AcousticCalculator
from core import DarkShipTracker

from hydrophone import Hydrophone
from ship import Ship


class SimulationManager:
    """Handles environment setup and configuration parsing"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.hydrophones: list[Hydrophone] = []
        self.ships: list[Ship] = []
        self.hydrophone_counter = 1  # Global hydrophone ID counter
        self.ship_counter = 1  # Global ship ID counter
        self.area = [0, 0, 0, 0]

    def _load_config(self, path: str):
        """Load simulation parameters from YAML file

        Args:
            path (str): Path to configuration file
        Returns:
            dict: Parsed configuration parameters
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def initialize_environment(self):
        """Create simulation entities from configuration"""

        # Set environment area
        self._set_area()

        # Create hydrophones
        self._create_manual_hydrophones()
        self._create_random_hydrophones()

        # Create ships
        self._create_manual_ships()
        self._create_random_ships()

        # Calculate hydrophones expected and observed acoustic pressures
        noise_level = self.config["hydrophones_config"].get("noise_level", 0)
        AcousticCalculator.calculate_pressures(
            self.hydrophones, self.ships, noise_level
        )

    def estimate_ds_positions(self):
        print(
            "Dark Ship triangulated position:", DarkShipTracker.mlat(self.hydrophones)
        )

    def _set_area(self):
        """Set area from configuration"""
        self.area = self.config.get(
            "area", [41, 42, 12, 13]
        )  # lat_min, lat_max, long_min, long_max

    def _create_manual_hydrophones(self):
        """Create hydrophone objects from configuration"""
        for hydro_data in self.config["hydrophones_config"].get("hydrophones", []):
            self.hydrophones.append(self._create_hydrophone_from_data(hydro_data))
            self.hydrophone_counter += 1

    def _create_hydrophone_from_data(self, hydrophone_data):
        """Create hydrophone from YAML configuration data

        Args:
            hydrophone_data (dict): Hydrophone parameters from YAML
        """
        return Hydrophone(
            id=self.hydrophone_counter,
            lat=hydrophone_data["coordinates"][0],
            long=hydrophone_data["coordinates"][1],
            depth=hydrophone_data["depth"],
            max_range=hydrophone_data["max_range"],
        )

    def _get_random_coordinates(self):
        lat_rand = np.random.uniform(self.area[0], self.area[1])
        long_rand = np.random.uniform(self.area[2], self.area[3])
        return [lat_rand, long_rand]

    def _create_random_hydrophones(self):
        # Random hydrophones
        num_random = self.config["hydrophones_config"].get("num_random", 0)
        max_range_range = self.config["hydrophones_config"].get(
            "max_range_range", [30, 50]
        )
        depth_range = self.config["hydrophones_config"].get("depth_range", [0, 0])

        for _ in range(num_random):
            self.hydrophones.append(
                self._create_random_hydrophone(max_range_range, depth_range)
            )
            self.hydrophone_counter += 1

    def _create_random_hydrophone(
        self, max_range_range: list[float], depth_range: list[float]
    ):
        """Generate random hydrophone within specified area"""
        lat, long = self._get_random_coordinates()
        max_range = np.random.uniform(max_range_range[0], max_range_range[1])
        depth = np.random.uniform(depth_range[0], depth_range[1])

        return Hydrophone(
            id=self.hydrophone_counter,
            lat=lat,
            long=long,
            max_range=max_range,
            depth=depth,
        )

    def _create_manual_ships(self):
        """Create manually defined ships from YAML configuration"""
        # Process AIS ships
        for ship_data in self.config["ships_config"].get("ais_ships", []):
            self.ships.append(self._create_ship_from_data(ship_data, False))
            self.ship_counter += 1

        # Process dark ships
        for ship_data in self.config["ships_config"].get("dark_ships", []):
            self.ships.append(self._create_ship_from_data(ship_data, True))
            self.ship_counter += 1

    def _create_random_ships(self):
        """Generate random ships to complete configured totals"""
        random_ais = self.config["ships_config"].get("num_random_ais_ships", 0)
        random_dark = self.config["ships_config"].get("num_random_dark_ships", 0)

        depth_range = self.config["ships_config"].get("depth_range", [0, 0])
        speed_range = self.config["ships_config"].get("speed_range", [10, 20])

        # Generate remaining random ships
        for _ in range(random_ais):
            self.ships.append(self._create_random_ship(speed_range, depth_range, False))
            self.ship_counter += 1

        for _ in range(random_dark):
            self.ships.append(self._create_random_ship(speed_range, depth_range, True))
            self.ship_counter += 1

    def _create_ship_from_data(self, ship_data, is_dark: bool):
        """Create ship from YAML configuration data

        Args:
            ship_data (dict): Ship parameters from YAML
            is_dark (bool): Dark ship status
        """
        return Ship(
            id=self.ship_counter,
            lat=ship_data["coordinates"][0],
            long=ship_data["coordinates"][1],
            depth=ship_data["depth"],
            speed=ship_data["speed"],
            is_dark=is_dark,
            base_pressure=ship_data.get(
                "base_pressure", 140 + 0.5 * ship_data["speed"]  # Default formula
            ),
        )

    def _create_random_ship(
        self, speed_range: list[float], depth_range: list[float], is_dark: bool
    ):
        """Generate random ship within configured parameters

        Args:
            is_dark (bool): Dark ship status
        """
        lat, long = self._get_random_coordinates()
        depth = np.random.uniform(depth_range[0], depth_range[1])
        speed = np.random.uniform(speed_range[0], speed_range[1])

        return Ship(
            id=self.ship_counter,
            lat=lat,
            long=long,
            depth=depth,
            speed=speed,
            is_dark=is_dark,
        )

    def plot_environment(self, map_ax):
        """Plot ships and hydrophones on a geographic map with proper legend handling."""
        margin = 1

        # Aggiungi il margine ai limiti esistenti
        lat_min = self.area[0] - margin
        lat_max = self.area[1] + margin
        long_min = self.area[2] - margin
        long_max = self.area[3] + margin

        # Crea il nuovo array dell'area con il margine
        plot_area = [long_min, long_max, lat_min, lat_max]

        map_ax.set_extent(plot_area, crs=ccrs.PlateCarree())
        map_ax.coastlines(resolution="110m")
        map_ax.add_feature(
            NaturalEarthFeature("physical", "land", "110m", edgecolor="black")
        )

        # -------------------------------------
        # |         Hydrophones plot          |
        # -------------------------------------
        hx = [
            h.coord.get_longitude() for h in self.hydrophones
        ]  # Lon, Lat instead of x, y
        hy = [h.coord.get_latitude() for h in self.hydrophones]  # Lat, Lon
        hydro_plot = map_ax.scatter(
            hx,
            hy,
            c="blue",
            marker="^",
            s=100,
            label="Hydrophones",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        hydro_labels = [
            f"Hydrophone {h.id}\n"
            f"Position: ({h.coord.get_latitude()}, {h.coord.get_longitude()})\n"
            f"Observed: {h.observed_pressure:.2f} dB\n"
            f"Expected: {h.expected_pressure:.2f} dB\n"
            f"Delta: {AcousticCalculator.compute_pressure_delta(h):.2f} dB"
            for h in self.hydrophones
        ]

        Utils.add_hover_tooltip(hydro_plot, hydro_labels)

        # -------------------------------------
        # |             Ships plot            |
        # -------------------------------------
        sx = [s.coord.get_longitude() for s in self.ships]  # Lon, Lat instead of x, y
        sy = [s.coord.get_latitude() for s in self.ships]  # Lat, Lon
        ship_colors = ["red" if s.is_dark else "green" for s in self.ships]

        ship_plot = map_ax.scatter(
            sx,
            sy,
            c=ship_colors,
            marker="o",
            s=150,
            label="Ships",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        ship_labels = [
            f"Ship {s.id}\n"
            f"Position: ({s.coord.get_latitude()}, {s.coord.get_longitude()})\n"
            f"Speed: {s.speed:.2f} knots\n"
            f"Base ac pressure: {s.base_pressure:.2f} dB\n"
            f"Is Dark: {s.is_dark}"
            for s in self.ships
        ]

        Utils.add_hover_tooltip(ship_plot, ship_labels)

        # Plot config
        map_ax.set_xlabel("Longitude", fontsize=12)
        map_ax.set_ylabel("Latitude", fontsize=12)
        map_ax.set_title("Ship and Hydrophone Locations", fontsize=14, pad=15)
        map_ax.grid(True, linestyle="--", alpha=0.6)

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Dark Ship",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=10,
                label="AIS Ship",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="Hydrophones",
            ),
        ]

        map_ax.legend(handles=legend_elements, loc="upper right")

    def plot_simulation(self):
        """Plot the environment with calculated statistics side by side."""
        if not self.hydrophones and not self.ships:
            print("The environment is empty!")
            return

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(1, 1)

        map_ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        self.plot_environment(map_ax)

        plt.tight_layout()
        plt.show()
