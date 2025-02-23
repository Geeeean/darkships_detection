import numpy as np
import matplotlib.pyplot as plt
import yaml

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


    def plot_environment(self):
        """Plot ships and hydrophones on a map with proper legend handling."""
        if not self.hydrophones and not self.ships:
            print("The environment is empty!")
            return

        _, ax = plt.subplots(figsize=(10, 8))

        # Hydrophones plot
        hx = [h.x for h in self.hydrophones]
        hy = [h.y for h in self.hydrophones]
        hydro_plot = ax.scatter(
            hx, hy,
            c='blue',
            marker='^',
            s=100,
            label='Hydrophones',
            zorder=3
        )

        # Ships plot
        ship_types = {}
        for ship in self.ships:
            color = 'red' if ship.is_dark else 'green'
            label = 'Dark Ship' if ship.is_dark else 'AIS Ship'

            if label not in ship_types:
                ship_types[label] = ax.scatter(
                    ship.x, ship.y,
                    c=color,
                    marker='o',
                    s=80,
                    label=label,
                    zorder=2
                )
            else:
                ax.scatter(ship.x, ship.y, c=color, marker='o', s=80, zorder=2)

        # Plot config
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_title("Simulation Map", fontsize=14, pad=15)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Legend
        legend_elements = [hydro_plot] + list(ship_types.values())
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            title="Legend"
        )

        plt.tight_layout()
        plt.show()
