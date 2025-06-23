import json
import copy
import os
import sys
import numpy as np
import yaml

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from utils import Utils
from environment import Environment, GeoBoundingBox

from hydrophone import Hydrophone
from ship import Ship
from bathymetry import Bathymetry

SIMULATION_FOLDER = "simulation"


class Simulation:
    """Handles environment setup and configuration parsing"""

    def __init__(self, config_path: str, delta_t_sec: float, iterations: int, num_threads: int):
        self.config = self._load_config(config_path)
        self.object_counter = 1  # Global ship and hydro counter
        self.delta_t_sec = delta_t_sec
        self.iterations = iterations

        # Thread setup
        self.num_threads = num_threads

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
        print(f"| Using {self.num_threads} threads for parallel execution")

        self.time_spent = 0
        self.toa_variance = self.config["environment"].get("toa_variance", [0])

        print(f"| Cloning config file into output folder for further analysis...")
        with open(f"{self.output}/{self.name}/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

        print("+ Setup ended correctly\n")

    def initialize_environment(self, toa_variance: float):
        """Initialize environment from configuration"""

        self.object_counter = 1

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
        bathymetry_path = self.config["environment"].get("bathymetry_path", None)
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

    @staticmethod
    def add_positioning_error(latitude, longitude, error_radius_m=10):
        """
        Adds random positioning error to coordinates

        Args:
            latitude: original latitude in degrees
            longitude: original longitude in degrees
            error_radius_m: radius of positioning error in meters (default: 10m)

        Returns:
            tuple: (perturbed_latitude, perturbed_longitude)
        """
        # Approximate conversions (valid for mid-latitudes)
        meters_per_degree_lat = 111320  # meters per degree of latitude

        # Generate random error in polar coordinates
        error_distance = np.random.uniform(0, error_radius_m)  # error distance [0, 10m]
        error_angle = np.random.uniform(0, 2 * np.pi)  # random angle

        # Convert to cartesian components (meters)
        error_x_m = error_distance * np.cos(error_angle)
        error_y_m = error_distance * np.sin(error_angle)

        # Convert meters to degrees
        error_lat_deg = error_y_m / meters_per_degree_lat

        # For longitude, consider current latitude
        meters_per_degree_lon = meters_per_degree_lat * np.cos(np.radians(latitude))
        error_lon_deg = error_x_m / meters_per_degree_lon

        # Apply error to coordinates
        perturbed_lat = latitude + error_lat_deg
        perturbed_lon = longitude + error_lon_deg

        return perturbed_lat, perturbed_lon

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

        hydrophones_info = []
        for h in self.environment.hydrophones:
             # Add positioning error (10 meters radius)
             perturbed_lat, perturbed_lon = Simulation.add_positioning_error(
                 h.coord.latitude,
                 h.coord.longitude,
                 error_radius_m=10
             )

             hydrophones_info.append({
                 "id": h.id,
                 "longitude": perturbed_lon,
                 "latitude": perturbed_lat,
                 "depth": h.coord.depth,
                 "observed_pressure": h.observed_pressure,
             })

        return {
            "ships": ships_info,
            "hydrophones": hydrophones_info,
            # "area": self.environment.area,
            "time_spent": self.time_spent,
        }

    def run_single_iteration(self, iteration_id: int, total_steps: int):
        """Run a single iteration of the simulation in a separate thread"""
        # Set unique seed for this iteration (thread-safe)
        np.random.seed(42 + iteration_id)

        # Array to collect all data for this iteration
        iteration_data = []

        for i in range(len(self.toa_variance)):
            # print(f"IT ID: [{iteration_id}], variance: {self.toa_variance[i]}")

            variance = self.toa_variance[i]

            # Initialize environment for this variance
            self.initialize_environment(variance)

            # Reset time
            self.time_spent = 0
            t = 0

            # Collect all data for this variance
            variance_data = []
            while t < total_steps:
                self.time_spent = t * self.delta_t_sec
                self.update_simulation(self.delta_t_sec)
                data = self.format_for_file()
                variance_data.append(copy.deepcopy(data))
                t += 1

            # Add to iteration array
            iteration_data.append({"variance": variance, "data": variance_data})

        return iteration_id, iteration_data

    def run(self, total_steps):
        """Run the simulation with parallel execution"""
        start_time = time.time()
        num_digits = len(str(self.iterations))
        out_folder = f"{self.output}/{self.name}/{SIMULATION_FOLDER}"

        print(f"+ Starting simulation")

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.num_threads, thread_name_prefix="SimWorker") as executor:
            # Submit all iterations
            future_to_iteration = {
                executor.submit(self.run_single_iteration, it, total_steps): it
                for it in range(self.iterations)
            }

            # Process completed iterations
            completed = 0
            for future in as_completed(future_to_iteration):
                iteration_id, iteration_data = future.result()

                # Write results to file
                iteration_str = str(iteration_id).zfill(num_digits)
                f_name = f"{iteration_str}_simulation.jsonl"
                out_name = f"{out_folder}/{f_name}"

                with open(out_name, "w") as f:
                    f.write(json.dumps(iteration_data) + "\n")

                completed += 1
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (self.iterations - completed) if completed > 0 else 0

                print(f"+ Completed iteration {iteration_id} ({completed}/{self.iterations})")

        total_time = time.time() - start_time
        print(f"\n+ All iterations completed in {total_time:.1f}s")
        print(f"| Average time per iteration: {total_time/self.iterations:.1f}s")


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
    threads = 1

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

    # Parse threads
    if "-t" in sys.argv:
        try:
            t_index = sys.argv.index("-t")
            threads = int(sys.argv[t_index + 1])
            if threads <= 0:
                print("Error: threads must be a positive number.")
                sys.exit(1)
        except (ValueError, IndexError):
            print("Error: -t option requires a valid integer.")
            sys.exit(1)

    return config_path, iterations, steps, threads


def main():
    """Main function for simulation module"""
    try:
        config_path, iterations, steps, threads = parse_args()

        threads = min(threads, os.cpu_count(), iterations)

        print(f"+ Starting Darkships Simulation")
        print(f"| Config: {config_path}")
        print(f"| Iterations: {iterations}")
        print(f"| Steps per iteration: {steps}")
        print(f"| Threads: {threads if threads else 'auto'}")

        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        # Create and run simulation
        sim = Simulation(config_path, 60, iterations, threads)
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
