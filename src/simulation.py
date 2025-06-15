import json
import os
import sys
import numpy as np
import yaml

from utils import Utils
from environment import Environment, GeoBoundingBox

from hydrophone import Hydrophone
from ship import Ship
from bathymetry import Bathymetry

SIMULATION_FOLDER = "simulation"


class Simulation:
    """Handles environment setup and configuration parsing"""

    def __init__(self, config_path: str, delta_t_sec: float, iterations: int):
        self.config = self._load_config(config_path)
        self.object_counter = 1  # Global ship and hydro counter
        self.delta_t_sec = delta_t_sec
        self.iterations = iterations

        self.setup_sim()

    def _load_config(self, path: str):
        """Load simulation parameters from YAML file

        Args:
            path (str): Path to configuration file
        Returns:
            dict: Parsed configuration parameters
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def setup_sim(self):
        """Setup simulation from config file"""
        self.name = self.config["name"]
        self.output = self.config["output_path"]

        folder = f"{self.output}/{self.name}"

        Utils.create_empty_folder(folder)
        Utils.create_empty_folder(f"{folder}/{SIMULATION_FOLDER}")
        print(f"| Writing in initialized output folder: {folder}")

        self.time_spent = 0
        self.toa_variance = self.config["environment"].get("toa_variance", [0])

        print(f"| Cloning config file into output folder for further analysis...")
        with open(f"{self.output}/{self.name}/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

        print("+ Setup ended correctly\n")

    def initialize_environment(self, toa_variance: float):
        """Initialize environment from configuration"""

        self.environment = Environment(
            area=self._get_area(),
            bathymetry=self._get_bathymetry(),
            noise_level=self._get_noise_level(),
            toa_variance=toa_variance,
        )

        # Create hydrophones
        self._create_manual_hydrophones()
        self._create_random_hydrophones()

        # Create ships
        self._create_manual_ships()
        self._create_random_ships()

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
            self.object_counter += 1

    def _create_hydrophone_from_data(self, hydrophone_data) -> Hydrophone:
        """Create hydrophone from YAML configuration data

        Args:
            hydrophone_data (dict): Hydrophone parameters from YAML
        """
        return Hydrophone(
            id=self.object_counter,
            lat=hydrophone_data["coordinates"][0],
            long=hydrophone_data["coordinates"][1],
            depth=hydrophone_data["depth"],
        )

    def _create_random_hydrophones(self):
        # Random hydrophones
        num_random = self.config["hydrophones_config"].get("num_random", 0)
        depth_range = self.config["hydrophones_config"].get("depth_range", [0, 0])

        for _ in range(num_random):
            hydrophone = self._create_random_hydrophone(depth_range)
            self.environment.add_hydrophone(hydrophone)
            self.object_counter += 1

    def _create_random_hydrophone(self, depth_range: list[float]):
        """Generate random hydrophone within specified area"""
        lat, long = self.environment.get_random_coordinates()
        depth = np.random.uniform(depth_range[0], depth_range[1])

        return Hydrophone(
            id=self.object_counter,
            lat=lat,
            long=long,
            depth=depth,
        )

    def _create_manual_ships(self):
        """Create manually defined ships from YAML configuration"""
        # Process AIS ships
        for ship_data in self.config["ships_config"].get("ais_ships", []):
            ship = self._create_ship_from_data(ship_data, False)
            self.environment.add_ship(ship)
            self.object_counter += 1

        # Process dark ships
        for ship_data in self.config["ships_config"].get("dark_ships", []):
            ship = self._create_ship_from_data(ship_data, True)
            self.environment.add_ship(ship)
            self.object_counter += 1

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
            self.object_counter += 1

        for _ in range(random_dark):
            ship = self._create_random_ship(speed_range, depth_range, True)
            self.environment.add_ship(ship)
            self.object_counter += 1

    def _create_ship_from_data(self, ship_data, is_dark: bool):
        """Create ship from YAML configuration data

        Args:
            ship_data (dict): Ship parameters from YAML
            is_dark (bool): Dark ship status
        """
        return Ship(
            id=self.object_counter,
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
            id=self.object_counter,
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

    def format_for_file(self):
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
                "depth": h.coord.depth,
                "observed_pressure": h.observed_pressure,
                # "expected_pressure": h.expected_pressure,
            }
            for h in self.environment.hydrophones
        ]

        return {
            "ships": ships_info,
            "hydrophones": hydrophones_info,
            "area": self.environment.area,
            "time_spent": self.time_spent,
        }

    def run(self, total_steps):
        """Run the simulation"""
        num_digits = len(str(self.iterations))
        out_folder = f"{self.output}/{self.name}/{SIMULATION_FOLDER}"

        for it in range(self.iterations):
            print(f"+ Computing iteration {it}")

            # Nome file unico per questa iterazione
            iteration_str = str(it).zfill(num_digits)
            f_name = f"{iteration_str}_simulation.jsonl"
            out_name = f"{out_folder}/{f_name}"

            print(f"| Producing file {f_name}")

            # Array per raccogliere tutti i dati di questa iterazione
            iteration_data = []

            for i in range(len(self.toa_variance)):
                variance = self.toa_variance[i]
                print(f"| Processing variance {variance}")

                self.initialize_environment(variance)
                self.time_spent = 0
                t = 0

                # Raccogli tutti i dati per questa varianza
                variance_data = []
                while t < total_steps:
                    self.time_spent = t * self.delta_t_sec
                    self.update_simulation(t * self.delta_t_sec)
                    data = self.format_for_file()
                    variance_data.append(data)
                    t += 1

                # Aggiungi all'array dell'iterazione
                iteration_data.append({"variance": variance, "data": variance_data})

            # Scrivi tutto l'array nel file
            with open(out_name, "w") as f:
                f.write(json.dumps(iteration_data) + "\n")

            print(f"+ Iteration {it} ended correctly\n")


def parse_args():
    """Parse command line arguments for simulation"""
    if len(sys.argv) < 2:
        print("Error: You must specify the path of the config file.")
        print(
            "Usage: python simulation.py /path/to/config.yaml [-i <iterations>] [-s <steps>]"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    iterations = 1
    steps = 5

    # Parse iterations
    if "-i" in sys.argv:
        try:
            i_index = sys.argv.index("-i")
            if i_index + 1 >= len(sys.argv):
                print("Error: -i option requires a number.")
                sys.exit(1)

            iterations = int(sys.argv[i_index + 1])
            if iterations <= 0:
                print("Error: iterations must be a positive number.")
                sys.exit(1)

        except ValueError:
            print("Error: iterations must be a valid integer.")
            sys.exit(1)

    # Parse steps
    if "-s" in sys.argv:
        try:
            s_index = sys.argv.index("-s")
            if s_index + 1 >= len(sys.argv):
                print("Error: -s option requires a number.")
                sys.exit(1)

            steps = int(sys.argv[s_index + 1])
            if steps <= 0:
                print("Error: steps must be a positive number.")
                sys.exit(1)

        except ValueError:
            print("Error: steps must be a valid integer.")
            sys.exit(1)

    return config_path, iterations, steps


def main():
    """Main function for simulation module"""
    try:
        config_path, iterations, steps = parse_args()

        print(f"+ Starting Darkships Simulation")
        print(f"| Config: {config_path}")
        print(f"| Iterations: {iterations}")
        print(f"| Steps per iteration: {steps}")

        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        # Create and run simulation
        sim = Simulation(config_path, 60, iterations)
        sim.run(steps)

        print("+ Simulation completed successfully!")

    except KeyboardInterrupt:
        print("+ Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"- Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# will produce...
# [
#   {
#     "variance": 1.0e-5,
#     "data": [
#       {"ships": [...], "hydrophones": [...], "time_spent": 0},
#       {"ships": [...], "hydrophones": [...], "time_spent": 60},
#       ...
#     ]
#   },
#   {
#     "variance": 1.0e-4,
#     "data": [
#       {"ships": [...], "hydrophones": [...], "time_spent": 0},
#       {"ships": [...], "hydrophones": [...], "time_spent": 60},
#       ...
#     ]
#   },
#   ...
# ]
