import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# === 1. Carica il file JSONL ===
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def plot_error(df):
    df["wcl_error"] = np.abs(df["ship_lat"] - df["centroid_lat"]) * 111320
    df["tdoa_error"] = np.abs(df["ship_lat"] - df["tdoa_lat"]) * 111320
    df["tmm_error"] = np.abs(df["ship_lat"] - df["tmm_lat"]) * 111320

    print(df["wcl_error"])
    print(df["tdoa_error"])
    print(df["tmm_error"])


    # === 4. Plot errori nel tempo ===
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["time"], df["wcl_error"], label="WCL Error", linestyle="--", color="blue"
    )
    plt.plot(
        df["time"], df["tdoa_error"], label="TDOA Error", linestyle="--", color="green"
    )
    plt.plot(
        df["time"], df["tmm_error"], label="TMM Error", linestyle="--", color="red"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Localization Latitude Error [deg]")
    plt.title("Localization Error Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    input_path = parse_args()
    data = load_jsonl(input_path)

    # === 2. Build DataFrame ===
    rows = []
    for entry in data:
        ship = entry["ships"][0]
        track = entry["tracking"]
        time = entry["time_spent"]

        rows.append(
            {
                "time": time,
                "ship_lat": ship["latitude"],
                "ship_lon": ship["longitude"],
                "centroid_lat": track.get("centroid", [None, None])[0],
                "centroid_lon": track.get("centroid", [None, None])[1],
                "tdoa_lat": track.get("tdoa", [None, None])[0],
                "tdoa_lon": track.get("tdoa", [None, None])[1],
                "tmm_lat": track.get("tmm", [None, None])[0],
                "tmm_lon": track.get("tmm", [None, None])[1],
            }
        )

    df = pd.DataFrame(rows)
    print(df)

    plot_error(df)


def parse_args():
    """Parse arguments and return input_path"""
    if len(sys.argv) < 1:
        print("Error: You must specify the path of the input file.")
        print("Usage: python main.py /path/to/file.jsonl")
        sys.exit(1)

    config_path = sys.argv[1]
    return config_path


if __name__ == "__main__":
    main()
