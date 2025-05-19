import json
import yaml

import numpy as np

from hydrophone import Hydrophone
from core import CoordinateHandler, Core
from utils import Utils


class Tracking:
    """Runs tracking algorithms on simulation output"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        path = self.config["output_path"]
        name = self.config["name"]
        self.path = f"{path}/{name}"

        self.sim_files = Utils._ls(self.path, "sim")

    def _load_config(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_hydrophones(self, snapshot: dict) -> list[Hydrophone]:
        """Ricrea la lista di hydrophones da una snapshot"""
        hydrophones = []
        for h_data in snapshot["hydrophones"]:
            h = Hydrophone(
                id=h_data["id"],
                lat=h_data["latitude"],
                long=h_data["longitude"],
                depth=h_data["depth"],
            )
            h.observed_pressure = h_data["observed_pressure"]
            hydrophones.append(h)
        return hydrophones

    def _format_for_file(self, snapshot: dict, tracking: dict):
        hydrophones_info = [
            {
                "id": h["id"],
                "longitude": h["longitude"],
                "latitude": h["latitude"],
                "depth": h["depth"],
                "observed_pressure": h["observed_pressure"][-1],
            }
            for h in snapshot["hydrophones"]
        ]

        ships_info = [
            {
                "id": s["id"],
                "longitude": s["longitude"],
                "latitude": s["latitude"],
                "is_dark": s["is_dark"],
                "heading": s["heading"],
                "speed": s["speed"],
            }
            for s in snapshot["ships"]
        ]

        return {
            "ships": ships_info,
            "hydrophones": hydrophones_info,
            "area": snapshot["area"],
            "time_spent": snapshot["time_spent"],
            "tracking": tracking,
        }

    def run(self):
        """Esegue il tracking per ogni snapshot del file input"""
        for sim_input in self.sim_files:
            f_name = sim_input.split("/")[-1]
            with open(sim_input, "r") as r:

                print(f"\n\nComputing tracking on file: {f_name}")
                out_file = f"{self.path}/tracking_{f_name}"
                with open(out_file, "w") as w:
                    print(f"Writing on file {out_file}")
                    first = True
                    for line in r:
                        if first:
                            first = False
                            continue

                        snapshot = json.loads(line)
                        hydrophones = self._load_hydrophones(snapshot)

                        d_ship = snapshot["ships"][0]
                        true_pos = (d_ship["latitude"], d_ship["longitude"])

                        centroid_pos = Core.weighted_centroid_localization(hydrophones)
                        tdoa_pos = Core.tdoa_localization(hydrophones)
                        tmm_pos = Core.tmm_localization(hydrophones)
                        sr_ls_pos = Core.sr_ls_localization(hydrophones)

                        print(
                            f"\nTime: {snapshot['time_spent']} s\t\tPosition\t\t\t\tError"
                        )
                        print(f" - Real position: \t{true_pos}\t0")
                        print(
                            f" - Weighted Centroid: \t{centroid_pos}\t{self.calculate_distance_error(hydrophones,centroid_pos, true_pos)}"
                        )
                        print(
                            f" - TDOA: \t\t{tdoa_pos}\t{self.calculate_distance_error(hydrophones,tdoa_pos, true_pos)}"
                        )
                        print(
                            f" - TMM: \t\t{tmm_pos}\t{self.calculate_distance_error(hydrophones,tmm_pos, true_pos)}"
                        )
                        print(
                            f" - SR-LS: \t\t{sr_ls_pos}\t{self.calculate_distance_error(hydrophones,sr_ls_pos, true_pos)}"
                        )

                        data = self._format_for_file(
                            snapshot,
                            {
                                "centroid": centroid_pos,
                                "tdoa": tdoa_pos,
                                "tmm": tmm_pos,
                                "sr_ls": sr_ls_pos,
                            },
                        )
                        w.write(json.dumps(data) + "\n")

    def calculate_distance_error(self, hydrophones, pos_estimated, pos_true):
        """Calculate squared error between estimated and true positions"""
        # Get UTM coordinates for accurate distance calculation
        utility_localizer = CoordinateHandler(hydrophones)

        # Convert positions to UTM
        e1, n1 = utility_localizer.geo_to_utm(pos_estimated[0], pos_estimated[1])
        e2, n2 = utility_localizer.geo_to_utm(pos_true[0], pos_true[1])

        # Calculate squared error (in square meters)
        distance_error = np.abs(e1 - e2)
        return distance_error
