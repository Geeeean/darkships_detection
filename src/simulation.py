import numpy as np
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
        # Calculate remaining AIS ships needed
        total_ais = self.config['ships'].get('num_ais_ships', 0)
        existing_ais = len([s for s in self.ships if not s.is_dark])
        remaining_ais = max(0, total_ais - existing_ais)

        # Calculate remaining dark ships needed
        total_dark = self.config['ships'].get('num_dark_ships', 0)
        existing_dark = len([s for s in self.ships if s.is_dark])
        remaining_dark = max(0, total_dark - existing_dark)

        # Generate remaining random ships
        for _ in range(remaining_ais):
            self.ships.append(self._create_random_ship(False))
            self.ship_counter += 1

        for _ in range(remaining_dark):
            self.ships.append(self._create_random_ship(True))
            self.ship_counter += 1

    def _create_ship_from_data(self, ship_data, is_dark):
        """Create ship from YAML configuration data

        Args:
            ship_data (dict): Ship parameters from YAML
            is_dark (bool): Dark ship status
        """
        return Ship(
            id=ship_data.get('id', self.ship_counter),
            x=ship_data['x'],
            y=ship_data['y'],
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
