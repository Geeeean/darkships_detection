import time
import json
import yaml

from hydrophone import Hydrophone
from core import Core
from utils import Utils


class Tracking:
    """Runs tracking algorithms on simulation output"""

    def __init__(self, output_path: str):
        self.config = self._load_config(f"{output_path}/config.yaml")

        path = self.config["output_path"]
        name = self.config["name"]
        self.path = f"{path}/{name}"

        self.sim_files = Utils._ls(self.path, "sim")

        print(self.sim_files)

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
        run_timestamp = str(int(time.time()))

        print(f"Reading and writing from/into {self.path}")

        for sim_input in self.sim_files:
            f_name = sim_input.split("/")[-1]
            out_file = f"{self.path}/tracking_{f_name}"

            print(f"Reading data from: {f_name} | Writing data in: tracking_{f_name}")

            # Raccogli tutti i dati in una lista
            run_data = []

            with open(sim_input, "r") as r:
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

                    data = self._format_for_file(
                        snapshot,
                        {
                            "truth_pos": true_pos,
                            "centroid": centroid_pos,
                            "tdoa": tdoa_pos,
                            "tmm": tmm_pos,
                            "sr_ls": sr_ls_pos,
                        },
                    )
                    run_data.append(data)

            # Crea il dizionario con timestamp
            timestamped_data = {run_timestamp: run_data}

            # Scrivi tutto insieme nel file
            with open(out_file, "a") as w:
                w.write(json.dumps(timestamped_data) + "\n")
