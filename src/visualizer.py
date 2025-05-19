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
    df["sr_ls_error"] = np.abs(df["ship_lat"] - df["sr_ls_lat"]) * 111320
    df["tdoa_error"] = np.abs(df["ship_lat"] - df["tdoa_lat"]) * 111320
    df["tmm_error"] = np.abs(df["ship_lat"] - df["tmm_lat"]) * 111320

    print(df["sr_ls_error"])
    print(df["tdoa_error"])
    print(df["sr_ls_error"])

    # === 4. Plot errori nel tempo ===
    plt.figure(figsize=(10, 6))
    plt.plot(df["time"], df["sr_ls_error"], label="SR-LS Error", color="blue")
    plt.plot(df["time"], df["tdoa_error"], label="TDOA Error", color="black")
    plt.plot(df["time"], df["tmm_error"], label="TMM Error", color="red")
    plt.xlabel("Time [s]")
    plt.ylabel("Localization x Error [m]")
    plt.title("Localization Error Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_environment(env_df):
    """
    Visualizza una mappa con le posizioni degli idrofoni e della nave
    """
    plt.figure(figsize=(12, 10))

    # Estrai i dati dell'ultimo timestamp
    latest_env = env_df.iloc[-1]

    # Plotta le posizioni degli idrofoni
    hydro_lats = []
    hydro_lons = []
    for hydro in latest_env["hydrophones"]:
        hydro_lats.append(hydro["hydro_lat"])
        hydro_lons.append(hydro["hydro_lon"])

    plt.scatter(
        hydro_lons, hydro_lats, color="blue", marker="^", s=100, label="Idrofoni"
    )

    # Aggiungi etichette agli idrofoni
    for i, (lon, lat) in enumerate(zip(hydro_lons, hydro_lats)):
        plt.annotate(
            f"H{i+1}",
            (lon, lat),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Plotta la posizione della nave
    ship_lats = env_df["ship_lat"].to_list()
    ship_lons = env_df["ship_lon"].to_list()

    # Plotta la traiettoria della nave
    plt.plot(ship_lons, ship_lats, "k-", label="Traiettoria nave")
    plt.scatter(ship_lons, ship_lats, color="black", s=20, alpha=0.5)

    # Evidenzia la posizione finale della nave
    plt.scatter(
        latest_env["ship_lon"],
        latest_env["ship_lat"],
        color="red",
        marker="*",
        s=200,
        label="Posizione finale nave",
    )

    # Calcola i confini della mappa
    all_lats = hydro_lats + ship_lats
    all_lons = hydro_lons + ship_lons

    # Calcola il centroide della mappa
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # Calcola la distanza massima dal centro
    max_dist_lat = max([abs(lat - center_lat) for lat in all_lats])
    max_dist_lon = max([abs(lon - center_lon) for lon in all_lons])

    # Aggiungi un margine del 20%
    margin = 0.2
    max_dist_lat = max_dist_lat * (1 + margin)
    max_dist_lon = max_dist_lon * (1 + margin)

    # Imposta i limiti della mappa
    plt.xlim(center_lon - max_dist_lon, center_lon + max_dist_lon)
    plt.ylim(center_lat - max_dist_lat, center_lat + max_dist_lat)

    # Calcola la scala approssimativa (1 grado di latitudine ≈ 111 km)
    scale_km = max_dist_lat * 111
    scale_text = f"Scala: {scale_km:.2f} km"

    # Aggiungi la scala
    plt.annotate(
        scale_text,
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
    )

    plt.grid(True)
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.title("Mappa delle posizioni degli idrofoni e della nave")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main():
    input_path = parse_args()
    data = load_jsonl(input_path)

    # === 2. Build DataFrames ===
    tracking_rows = []
    env_rows = []

    for entry in data:
        ship = entry["ships"][0]
        hydrophones = entry["hydrophones"]
        track = entry["tracking"]
        time = entry["time_spent"]

        tracking_rows.append(
            {
                "time": time,
                "ship_lat": ship["latitude"],
                "ship_lon": ship["longitude"],
                "sr_ls_lat": track.get("sr_ls", [None, None])[0],
                "sr_ls_lon": track.get("sr_ls", [None, None])[1],
                "tdoa_lat": track.get("tdoa", [None, None])[0],
                "tdoa_lon": track.get("tdoa", [None, None])[1],
                "tmm_lat": track.get("tmm", [None, None])[0],
                "tmm_lon": track.get("tmm", [None, None])[1],
            }
        )

        env_rows.append(
            {
                "time": time,
                "ship_lat": ship["latitude"],
                "ship_lon": ship["longitude"],
                "hydrophones": [
                    {"hydro_lat": h["latitude"], "hydro_lon": h["longitude"]}
                    for h in hydrophones
                ],
            }
        )

    tracking_df = pd.DataFrame(tracking_rows)
    env_df = pd.DataFrame(env_rows)

    plot_environment(env_df)
    plot_error(tracking_df)


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
