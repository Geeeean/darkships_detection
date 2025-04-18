from multiprocessing import Manager, Queue
import numpy as np
import yaml

from environment import Environment, GeoBoundingBox

from core import DarkShipTracker

from hydrophone import Hydrophone
from ship import Ship
from bathymetry import Bathymetry


class Simulation:
    """Handles environment setup and configuration parsing"""

    def __init__(self, config_path: str, read_queue: Queue, write_queue: Queue):
        self.config = self._load_config(config_path)
        self.hydrophone_counter = 1  # Global hydrophone ID counter
        self.ship_counter = 1  # Global ship ID counter
        self.read = read_queue
        self.write = write_queue
        self.status = "pause"

        self.initialize_environment()

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
        self.environment = Environment(
            area=self._get_area(),
            bathymetry=self._get_bathymetry(),
            noise_level=self._get_noise_level(),
        )

        # Create hydrophones
        self._create_manual_hydrophones()
        self._create_random_hydrophones()

        # Create ships
        self._create_manual_ships()
        self._create_random_ships()

    def estimate_ds_positions(self):
        est_pos = DarkShipTracker.mlat(self.environment)
        print("Dark Ship triangulated position:", est_pos)

    def _get_area(self):
        """Set area from configuration"""
        area: GeoBoundingBox = self.config["environment"].get("area")
        return area

    def _get_bathymetry(self):
        bathymetry_path = self.config["environment"].get("bathymetry_path")
        return Bathymetry(bathymetry_path)

    def _get_noise_level(self):
        noise_level = self.config["hydrophones_config"].get("noise_level", 0)
        return noise_level

    def _create_manual_hydrophones(self):
        """Create hydrophone objects from configuration"""
        for hydro_data in self.config["hydrophones_config"].get("hydrophones", []):
            hydrophone: Hydrophone = self._create_hydrophone_from_data(hydro_data)
            self.environment.add_hydrophone(hydrophone)
            self.hydrophone_counter += 1

    def _create_hydrophone_from_data(self, hydrophone_data) -> Hydrophone:
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

    def _create_random_hydrophones(self):
        # Random hydrophones
        num_random = self.config["hydrophones_config"].get("num_random", 0)
        max_range_range = self.config["hydrophones_config"].get(
            "max_range_range", [30, 50]
        )
        depth_range = self.config["hydrophones_config"].get("depth_range", [0, 0])

        for _ in range(num_random):
            hydrophone = self._create_random_hydrophone(max_range_range, depth_range)
            self.environment.add_hydrophone(hydrophone)
            self.hydrophone_counter += 1

    def _create_random_hydrophone(
        self, max_range_range: list[float], depth_range: list[float]
    ):
        """Generate random hydrophone within specified area"""
        lat, long = self.environment.get_random_coordinates()
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
            ship = self._create_ship_from_data(ship_data, False)
            self.environment.add_ship(ship)
            self.ship_counter += 1

        # Process dark ships
        for ship_data in self.config["ships_config"].get("dark_ships", []):
            ship = self._create_ship_from_data(ship_data, True)
            self.environment.add_ship(ship)
            self.ship_counter += 1

    def _create_random_ships(self):
        """Generate random ships to complete configured totals"""
        random_ais = self.config["ships_config"].get("num_random_ais_ships", 0)
        random_dark = self.config["ships_config"].get("num_random_dark_ships", 0)

        depth_range = self.config["ships_config"].get("depth_range", [0, 0])
        speed_range = self.config["ships_config"].get("speed_range", [10, 20])

        # Generate remaining random ships
        for _ in range(random_ais):
            ship = self._create_random_ship(speed_range, depth_range, False)
            self.environment.add_ship(ship)
            self.ship_counter += 1

        for _ in range(random_dark):
            ship = self._create_random_ship(speed_range, depth_range, True)
            self.environment.add_ship(ship)
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
            heading=ship_data["heading"],
        )

    def _create_random_ship(
        self, speed_range: list[float], depth_range: list[float], is_dark: bool
    ):
        """Generate random ship within configured parameters

        Args:
            is_dark (bool): Dark ship status
        """
        lat, long = self.environment.get_random_coordinates()
        depth = np.random.uniform(depth_range[0], depth_range[1])
        speed = np.random.uniform(speed_range[0], speed_range[1])
        heading = np.random.uniform(0, 360)

        return Ship(
            id=self.ship_counter,
            lat=lat,
            long=long,
            depth=depth,
            speed=speed,
            is_dark=is_dark,
            heading=heading,
        )

    def update_simulation(self, t: float):
        """Update simulation for one timestep"""
        # 1. Update ship positions
        for ship in self.environment.ships:
            if ship.is_dark:
                ship.update_position(t)

        # 2. Compute hydrophone pressures
        self.environment.calculate_pressures()

    def format_for_queue(self):
        ships_info = [
            {
                "id": s.id,
                "longitude": s.coord.longitude,
                "latitude": s.coord.latitude,
                "is_dark": s.is_dark,
                "heading": s.heading,
                "speed": s.speed,
            }
            for s in self.environment.ships
        ]

        hydrophones_info = [
            {
                "id": h.id,
                "longitude": h.coord.longitude,
                "latitude": h.coord.latitude,
                "observed_pressure": h.observed_pressure,
            }
            for h in self.environment.hydrophones
        ]

        return {
            "ships": ships_info,
            "hydrophones": hydrophones_info,
            "area": self.environment.area,
            "status": self.status,
        }

    def run(self, total_steps, delta_t_sec):
        """Run the simulation"""
        time_spent = 0
        self.initialize_environment()

        data = self.format_for_queue()
        self.write.put(data)

        while self.read.get() != "START":
            pass

        self.status = "run"

        t = 0
        while t < total_steps:
            if not self.read.empty():
                command = self.read.get_nowait()

                if command == "PAUSE":
                    self.status = "pause"

                    data = self.format_for_queue()
                    self.write.put(data)

                    print("[SIM] Pausing simulation")
                    while command != "START":
                        if not self.read.empty():
                            command = self.read.get_nowait()
                            if command == "RESTART":
                                print("[SIM] Restarting simulation from pause")
                                self.initialize_environment()
                                time_spent = 0
                                t = 0
                                data = self.format_for_queue()
                                self.write.put(data)
                                continue
                elif command == "RESTART":
                    print("[SIM] Restarting simulation")
                    self.initialize_environment()
                    time_spent = 0
                    t = 0
                    data = self.format_for_queue()
                    self.write.put(data)
                    continue

            self.status = "run"
            print(f"[SIM] Time elapsed {time_spent}s")
            time_spent += delta_t_sec

            self.update_simulation(t * delta_t_sec)

            data = self.format_for_queue()
            self.write.put(data)

            t += 1
