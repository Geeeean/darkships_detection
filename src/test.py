import json
import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from tracking import Tracking
from utils import Utils


def parse_args():
    """Parse arguments and return config_path"""
    if len(sys.argv) < 2:
        print("Error: You must specify the path of the config file.")
        print("Usage: python test.py /path/to/file.yaml")
        sys.exit(1)
    config_path = sys.argv[1]
    return config_path


def load_config(path: str):
    """Load simulation parameters from YAML file
    Args:
        path (str): Path to configuration file
    Returns:
        dict: Parsed configuration parameters
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    iterations = 100
    config_path = parse_args()
    config = load_config(config_path)

    # Extract folder and variance information
    folder = f"{config['output_path']}/{config['name']}"
    toa_variances = config["environment"].get("toa_variance", [])
    n = len(toa_variances)

    if n == 0:
        print("Error: No TOA variance values specified in config.")
        sys.exit(1)

    print(f"Running {iterations} iterations for {n} variance levels: {toa_variances}")

    # Create results directory if it doesn't exist
    os.makedirs(f"{folder}/results", exist_ok=True)

    # Initialize dictionaries to store latitude errors for each algorithm at each variance level
    # Structure: {variance_index: {algorithm: [errors_list]}}
    all_lat_errors = {i: {"tmm": [], "tdoa": [], "sr_ls": []} for i in range(n)}

    for iteration in range(iterations):
        print(f"Running iteration {iteration+1}/{iterations}")

        # Run the simulation
        sim = Simulation(config_path, 60)
        sim.run(n)  # Run once for each variance level

        # Run the tracking algorithms
        tracker = Tracking(config_path)
        tracker.run()

        # Process results for each variance level
        for variance_idx in range(n):
            result_file = f"{folder}/tracking_sim_{variance_idx}.jsonl"

            try:
                # Load the JSONL file for this variance level
                data = load_jsonl(result_file)

                # Process each entry in the JSONL file
                for entry in data:
                    hydrophones = tracker._load_hydrophones(entry)

                    track = entry["tracking"]
                    ship = entry["ships"][0]

                    # Extract latitudes
                    ship_pos = [float(ship["latitude"]), float(ship["longitude"])]

                    tmm_pos = track.get("tmm", [None, None])
                    tdoa_pos = track.get("tdoa", [None, None])
                    sr_ls_pos = track.get("sr_ls", [None, None])

                    # Calculate latitude errors if coordinates exist
                    if tmm_pos is not None:
                        tmm_lat_error = tracker.calculate_distance_error(
                            hydrophones, tmm_pos, ship_pos
                        )
                        all_lat_errors[variance_idx]["tmm"].append(tmm_lat_error)

                    if tdoa_pos is not None:
                        tdoa_lat_error = tracker.calculate_distance_error(
                            hydrophones, tdoa_pos, ship_pos
                        )
                        all_lat_errors[variance_idx]["tdoa"].append(tdoa_lat_error)

                    if sr_ls_pos is not None:
                        sr_ls_lat_error = tracker.calculate_distance_error(
                            hydrophones, sr_ls_pos, ship_pos
                        )
                        all_lat_errors[variance_idx]["sr_ls"].append(sr_ls_lat_error)

            except Exception as e:
                print(f"Error processing file {result_file}: {e}")

    # Calculate MSE for latitude errors for each algorithm at each variance level
    mse_results = {}
    for variance_idx, algorithm_errors in all_lat_errors.items():
        variance_value = toa_variances[variance_idx]
        mse_results[variance_value] = {}

        for algorithm, errors in algorithm_errors.items():
            if errors:  # Only calculate if we have errors
                mse = np.mean(np.square(errors))
                mean_error = np.mean(errors)
                std_error = np.std(errors)

                mse_results[variance_value][algorithm] = {
                    "mse": mse,
                    "mean_error": mean_error,
                    "std_error": std_error,
                    "num_samples": len(errors),
                }

    # Print results
    print("\nMSE x est Results by Variance Level:")
    print("-------------------------------------")
    for variance, algorithms in mse_results.items():
        print(f"\nVariance: {variance}")
        for algorithm, stats in algorithms.items():
            print(f"  {algorithm.upper()}:")
            print(f"    MSE x est: {stats['mse']:.2f} m²")
            print(f"    Mean x est Error: {stats['mean_error']:.2f} m")
            print(f"    Std Dev: {stats['std_error']:.2f} m")
            print(f"    Samples: {stats['num_samples']}")

    Utils.create_empty_folder(f"{folder}/results")

    # Save results to JSON file
    results_file = f"{folder}/results/latitude_mse_analysis.json"
    with open(results_file, "w") as f:
        json.dump(mse_results, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")

    # Create visualization
    plot_mse_comparison(mse_results, f"{folder}/results/latitude_mse_comparison.png")


def plot_mse_comparison(mse_results, output_file):
    """
    Create a log-log plot comparing latitude MSE across algorithms and variance levels.
    """
    plt.figure(figsize=(12, 8))

    # Prepare data for plotting
    variances = sorted(list(mse_results.keys()), key=float)
    algorithms = ["tmm", "tdoa", "sr_ls"]
    colors = ["red", "black", "blue"]

    for algorithm, color in zip(algorithms, colors):
        mse_values = []
        for variance in variances:
            if algorithm in mse_results[variance]:
                mse_values.append(mse_results[variance][algorithm]["mse"])
            else:
                mse_values.append(np.nan)  # Use NaN for missing values

        # Plot non-NaN values
        valid_indices = ~np.isnan(mse_values)
        valid_variances = [
            float(variances[i]) for i in range(len(variances)) if valid_indices[i]
        ]
        valid_mse = [mse_values[i] for i in range(len(mse_values)) if valid_indices[i]]

        if valid_variances:
            plt.loglog(
                valid_variances,
                valid_mse,
                "o-",
                label=algorithm.upper(),
                color=color,
                linewidth=2,
                markersize=8,
            )

    plt.xlabel("TOA Variance", fontsize=12)
    plt.ylabel("MSE x est (m²)", fontsize=12)
    plt.title("Algorithm Est Error Comparison Across Noise Levels", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    # Also create a bar chart for easier comparison
    create_bar_chart(mse_results, output_file.replace(".png", "_bar.png"))


def create_bar_chart(mse_results, output_file):
    """
    Create a bar chart comparing latitude MSE for algorithms at each variance level.
    """
    plt.figure(figsize=(14, 8))

    # Prepare data
    variances = sorted(list(mse_results.keys()), key=float)
    algorithms = ["tmm", "tdoa", "sr_ls"]
    colors = ["red", "black", "blue"]

    # Set positions for bars
    bar_width = 0.25
    r1 = np.arange(len(variances))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create bars
    for i, (algorithm, color, positions) in enumerate(
        zip(algorithms, colors, [r1, r2, r3])
    ):
        mse_values = []
        for variance in variances:
            if algorithm in mse_results[variance]:
                mse_values.append(mse_results[variance][algorithm]["mse"])
            else:
                mse_values.append(0)  # Use 0 for missing values in bar chart

        plt.bar(
            positions,
            mse_values,
            width=bar_width,
            color=color,
            edgecolor="grey",
            label=algorithm.upper(),
        )

    # Add labels, title, etc.
    plt.xlabel("TOA Variance", fontsize=12)
    plt.ylabel("MSE x est (m²)", fontsize=12)
    plt.yscale("log")  # Log scale for better visualization
    plt.title("MSE x Comparison by Algorithm and Variance Level", fontsize=14)
    plt.xticks(
        [r + bar_width for r in range(len(variances))],
        [f"{float(v):.1e}" for v in variances],
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    main()
