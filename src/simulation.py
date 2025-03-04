import numpy as np
import matplotlib.pyplot as plt
import yaml

from utils import Utils
from core import AcousticCalculator
from core import DarkShipTracker

class Hydrophone:
    """Represents an underwater acoustic sensor
    Attributes:
        id (int): Unique identifier
        x (float): X coordinate [meters]
        y (float): Y coordinate [meters]
        observed_noise (float): Measured noise level [dB re 1μPa]
        expected_noise (float): Predicted noise level from AIS data
    """
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.observed_noise = 0.0
        self.expected_noise = 0.0

class Ship:
    """Represents a vessel with acoustic properties
    Attributes:
        id (int): Unique identifier
        x (float): X coordinate [meters]
        y (float): Y coordinate [meters]
        speed (float): Speed [knots]
        is_dark (bool): True if not transmitting AIS
        base_noise (float): Acoustic signature at 1m distance
    """
    def __init__(self, id, x, y, speed, is_dark=False, base_noise = 140):
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed
        self.is_dark = is_dark
        self.base_noise = base_noise + 0.5 * speed  # Empirical noise-speed relationship

class SimulationManager:
    """Handles environment setup and configuration parsing"""
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.hydrophones = []
        self.ships = []
        self.hydrophone_counter = 1  # Global hydrophone ID counter
        self.ship_counter = 1  # Global ship ID counter
        self.area = [0, 0, 0, 0]

    def _load_config(self, path):
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

        # Calculate hydrophones noises
        AcousticCalculator.calculate_noises(self.hydrophones, self.ships, self.config)

    def estimate_ds_positions(self):
        print("Dark Ship triangulated position:", DarkShipTracker.triangulate_ship_position(self.hydrophones))

    def _set_area(self):
        """Set area from configuration"""
        self.area = self.config['hydrophones'].get('area', [0, 1000, 0, 1000])  # x_min, x_max, y_min, y_max

    def _create_manual_hydrophones(self):
        """Create hydrophone objects from configuration"""
        for pos in self.config['hydrophones'].get('positions', []):
            self.hydrophones.append(
                Hydrophone(self.hydrophone_counter, pos[0], pos[1])
            )
            self.hydrophone_counter += 1

    def _create_random_hydrophones(self):
        # Random hydrophones
        num_random = self.config['hydrophones'].get('num_random', 0)

        for _ in range(num_random):
            self.hydrophones.append(self._create_random_hydrophone())
            self.hydrophone_counter += 1

    def _create_random_hydrophone(self):
        """Generate random hydrophone within specified area"""
        x = np.random.uniform(self.area[0], self.area[1])
        y = np.random.uniform(self.area[2], self.area[3])
        return Hydrophone(self.hydrophone_counter, x, y)

    def _create_manual_ships(self):
        """Create manually defined ships from YAML configuration"""
        # Process AIS ships
        for ship_data in self.config['ships'].get('ais_ships', []):
            self.ships.append(self._create_ship_from_data(ship_data, False))
            self.ship_counter += 1

        # Process dark ships
        for ship_data in self.config['ships'].get('dark_ships', []):
            self.ships.append(self._create_ship_from_data(ship_data, True))
            self.ship_counter += 1

    def _create_random_ships(self):
        """Generate random ships to complete configured totals"""
        random_ais = self.config['ships'].get('num_random_ais_ships', 0)
        random_dark = self.config['ships'].get('num_random_dark_ships', 0)

        # Generate remaining random ships
        for _ in range(random_ais):
            self.ships.append(self._create_random_ship(False))
            self.ship_counter += 1

        for _ in range(random_dark):
            self.ships.append(self._create_random_ship(True))
            self.ship_counter += 1

    def _create_ship_from_data(self, ship_data, is_dark):
        """Create ship from YAML configuration data

        Args:
            ship_data (dict): Ship parameters from YAML
            is_dark (bool): Dark ship status
        """
        return Ship(
            id= self.ship_counter,
            x=ship_data['position'][0],
            y=ship_data['position'][1],
            speed=ship_data['speed'],
            is_dark=is_dark,
            base_noise=ship_data.get(
                'base_noise',
                140 + 0.5 * ship_data['speed']  # Default formula
            )
        )

    def _create_random_ship(self, is_dark):
        """Generate random ship within configured parameters

        Args:
            is_dark (bool): Dark ship status
        """
        return Ship(
            id=self.ship_counter,
            x=np.random.uniform(self.area[0], self.area[1]),
            y=np.random.uniform(self.area[2], self.area[3]),
            speed=np.random.uniform(*self.config['ships']['speed_range']),
            is_dark=is_dark
        )

    def plot_environment(self, map):
        """Plot ships and hydrophones on a map with proper legend handling."""

        # -------------------------------------
        # |         Hydrophones plot          |
        # -------------------------------------
        hx = [h.x for h in self.hydrophones]
        hy = [h.y for h in self.hydrophones]
        hydro_plot = map.scatter(
            hx, hy,
            c='blue',
            marker='^',
            s=100,
            label='Hydrophones',
            zorder=3
        )

        hydro_labels = [
            f"Hydrophone {h.id}\n"
                f"Position: ({h.x}, {h.y})\n"
                f"Observed: {h.observed_noise:.2f} dB\n"
                f"Expected: {h.expected_noise:.2f} dB\n"
                f"Delta: {AcousticCalculator.compute_noise_delta(h):.2f} dB"
            for h in self.hydrophones
        ]

        Utils.add_hover_tooltip(hydro_plot, hydro_labels)

        # -------------------------------------
        # |             Ships plot            |
        # -------------------------------------
        sx = [s.x for s in self.ships]
        sy = [s.y for s in self.ships]
        ship_colors = ['red' if s.is_dark else 'green' for s in self.ships]

        ship_plot = map.scatter(
            sx, sy,
            c=ship_colors,
            marker='o',
            s=150,
            label='Ships',
            zorder=3
        )

        ship_labels = [
            f"Ship {s.id}\n"
            f"Position: ({s.x}, {s.y})\n"
            f"Speed: {s.speed:.2f} knots\n"
            f"Base noise: {s.base_noise:.2f} dB\n"
            f"Is Dark: {s.is_dark}"
            for s in self.ships
        ]

        Utils.add_hover_tooltip(ship_plot, ship_labels)

        # Plot config
        map.set_xlabel("X (m)", fontsize=12)
        map.set_ylabel("Y (m)", fontsize=12)
        map.set_title("Simulation Map", fontsize=14, pad=15)
        map.grid(True, linestyle='--', alpha=0.6)

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Dark Ship'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='AIS Ship'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Hydrophones'),
        ]

        plt.legend(handles=legend_elements, loc='upper right')

    def plot_simulation(self):
        """Plot the environment with calculated statistics side by side."""
        if not self.hydrophones and not self.ships:
            print("The environment is empty!")
            return

        fig = plt.figure(figsize=(14, 8))
        #gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])  # 3:1 ratio for map:data
        gs = fig.add_gridspec(1, 1)

        map = fig.add_subplot(gs[0])
        self.plot_environment(map)

        #data = fig.add_subplot(gs[1])
        #self.plot_data(data)

        plt.tight_layout()
        plt.show()
