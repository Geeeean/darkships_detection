import json
import yaml
import sys
import os
from hydrophone import Hydrophone
from core import Core
from simulation import SIMULATION_FOLDER
from utils import Utils

TRACKING_FOLDER = "tracking"


class Tracking:
    """Runs tracking algorithms on simulation output"""

    def __init__(self, output_path: str):
        self.config = self._load_config(f"{output_path}/config.yaml")
        path = self.config["output_path"]
        name = self.config["name"]
        self.path = f"{path}/{name}"
        self.sim_files = Utils._ls(f"{self.path}/{SIMULATION_FOLDER}", None)

        print(f"| Reading and writing from/into {self.path}")
        print(f"| Found {len(self.sim_files)} simulation files")
        Utils.create_empty_folder(f"{self.path}/{TRACKING_FOLDER}")

        print("+ Setup ended correctly\n")

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
            f_number = f_name.split("_")[0]
            out_file = f"{f_number}_tracking.jsonl"
            out_path = f"{self.path}/{TRACKING_FOLDER}/{out_file}"

            # Raccogli tutti i dati in una lista
            write_data = []

            try:
                with open(sim_input, "r") as r:
                    print(f"+ Processing file {f_name}")
                    # Leggi l'intero contenuto del file (una riga con array JSON)
                    content = r.read().strip()
                    if not content:
                        print(f"| Warning: Empty file {f_name}")
                        continue

                    # Parse dell'array di simulazioni
                    simulations = json.loads(content)

                    # Itera attraverso ogni simulazione (oggetto con variance e data)
                    for simulation in simulations:
                        run_data = []
                        variance = simulation["variance"]
                        snapshots = simulation["data"]

                        print(
                            f"| Processing variance {variance} with {len(snapshots)} snapshots"
                        )

                        # Processa ogni snapshot in questa simulazione
                        for snapshot in snapshots:
                            hydrophones = self._load_hydrophones(snapshot)

                            # Trova la nave dark
                            dark_ships = [s for s in snapshot["ships"] if s["is_dark"]]
                            if not dark_ships:
                                print(
                                    f"| Warning: No dark ships found in snapshot at time {snapshot['time_spent']}"
                                )
                                continue

                            d_ship = dark_ships[0]
                            true_pos = (d_ship["latitude"], d_ship["longitude"])

                            try:
                                centroid_pos = Core.weighted_centroid_localization(
                                    hydrophones
                                )
                                tdoa_pos = Core.tdoa_localization(hydrophones)
                                tmm_pos = Core.tmm_localization(hydrophones)
                                sr_ls_pos = Core.sr_ls_localization(hydrophones)
                                tmm_sr_ls_pos = Core.tmm_localization_sr_ls_init(
                                    hydrophones
                                )

                                data = self._format_for_file(
                                    snapshot,
                                    {
                                        "truth_pos": true_pos,
                                        "centroid": centroid_pos,
                                        "tdoa": tdoa_pos,
                                        "tmm": tmm_pos,
                                        "sr_ls": sr_ls_pos,
                                        "tmm_sr_ls": tmm_sr_ls_pos,
                                    },
                                )

                                run_data.append(data)

                            except Exception as e:
                                print(
                                    f"+ Warning: Error processing snapshot at time {snapshot['time_spent']}: {e}\n"
                                )
                                continue

                        write_data.append({"data": run_data, "variance": variance})

                    # Scrivi tutto insieme nel file
                    with open(out_path, "a") as w:
                        w.write(json.dumps(write_data) + "\n")

                print(f"| Saved {len(run_data)} tracking results")
                print(f"+ File {f_name} processed successfully\n")

            except Exception as e:
                print(f"| Error processing file {f_name}: {e}")
                continue

        print(f"+ Tracking completed successfully!")


def parse_args():
    """Parse command line arguments for tracking"""
    if len(sys.argv) < 2:
        print("Error: You must specify the output folder path.")
        print("Usage: python tracking.py /path/to/output/folder")
        print("Example: python tracking.py output/sample_simulation")
        sys.exit(1)

    output_folder = sys.argv[1]
    return output_folder


def main():
    """Main function for tracking module"""
    try:
        print("+ Starting Tracking")
        output_folder = parse_args()

        # Check if output folder exists
        if not os.path.exists(output_folder):
            print(f"| Error: Output folder not found: {output_folder}")
            sys.exit(1)

        # Check if it's a directory
        if not os.path.isdir(output_folder):
            print(f"| Error: Path is not a directory: {output_folder}")
            sys.exit(1)

        # Check if config exists
        config_path = f"{output_folder}/config.yaml"
        if not os.path.exists(config_path):
            print(f"| Error: Config file not found: {config_path}")
            sys.exit(1)

        # Create and run tracking
        tracker = Tracking(output_folder)

        tracker.run()

    except KeyboardInterrupt:
        print("\n+ Tracking interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"| Error: Tracking failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# [
#   {
#     "data": [
#       {
#         "ships": [...],
#         "hydrophones": [...],
#         "area": [...],
#         "time_spent": 0,
#         "tracking": {
#           "truth_pos": [11.032, 84.015],
#           "centroid": [10.924202917740102, 83.9728944697951],
#           "tdoa": [11.005664581207112, 84.00508367181102],
#           "tmm": [11.007843804451092, 84.00759253461663],
#           "sr_ls": [11.005948131814343, 84.00538395289065]
#         }
#       },
#       // ... altri timestep per questa varianza
#     ],
#     "variance": 1e-05
#   },
#   // ... altri blocchi per diverse varianze
# ]
