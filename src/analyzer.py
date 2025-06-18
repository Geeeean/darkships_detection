import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import yaml
from typing import Dict, List, Tuple, Any
import warnings
from utils import Utils
from geopy.distance import geodesic

warnings.filterwarnings("ignore")


class TrackingAnalyzer:
    """Analyzer for tracking algorithm performance"""

    def __init__(self, output_folder: str):
        """
        Initialize analyzer with output folder containing tracking results

        Args:
            output_folder: Path to folder containing tracking/ subfolder
        """
        self.config = self._load_config(f"{output_folder}/config.yaml")
        path = self.config["output_path"]
        name = self.config["name"]
        self.path = f"{path}/{name}"

        self.tracking_folder = f"{self.path}/tracking"
        self.results_folder = f"{self.path}/analysis"

        print(f"| Reading tracking data from: {self.tracking_folder}")
        print(f"| Analysis results will be saved to: {self.results_folder}")

        # Create results folder using Utils
        Utils.create_empty_folder(self.results_folder)

        # Data storage
        self.tracking_data = {}
        self.processed_data = None
        self.algorithm_names = ["centroid", "tdoa", "tmm", "sr_ls"]

        print("+ Setup ended correctly\n")

    def _load_config(self, path: str):
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_tracking_data(self):
        """Load all tracking files from tracking folder"""
        print(f"+ Loading tracking data")

        if not os.path.exists(self.tracking_folder):
            raise FileNotFoundError(
                f"Tracking folder not found: {self.tracking_folder}"
            )

        # Find all tracking files
        tracking_files = []
        for filename in os.listdir(self.tracking_folder):
            if filename.endswith("_tracking.jsonl"):
                tracking_files.append(os.path.join(self.tracking_folder, filename))

        if not tracking_files:
            raise FileNotFoundError(
                f"No tracking files found in {self.tracking_folder}"
            )

        print(f"| Found {len(tracking_files)} tracking files")

        for tracking_file in tracking_files:
            filename = os.path.basename(tracking_file)
            file_key = filename.replace("_tracking.jsonl", "")  # e.g., "0"

            with open(tracking_file, "r") as f:
                # Each line contains an array of variance blocks
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            # Parse the array of variance blocks
                            variance_blocks = json.loads(line)

                            # variance_blocks is like:
                            # [
                            #   {"data": [...], "variance": 1e-05},
                            #   {"data": [...], "variance": 0.0001},
                            #   ...
                            # ]

                            if file_key not in self.tracking_data:
                                self.tracking_data[file_key] = []

                            # Just append the variance blocks (no timestamp)
                            self.tracking_data[file_key].extend(variance_blocks)

                        except json.JSONDecodeError as e:
                            print(
                                f"| Warning: Error parsing line {line_num + 1} in {filename}: {e}"
                            )

            print(
                f"| Loaded {len(self.tracking_data[file_key])} variance levels for {filename}"
            )

        print(f"| Loaded {len(self.tracking_data)} tracking file(s)")

        print(f"+ Data loading completed\n")

    def _calculate_distance_error_meters(self, true_lat, true_lon, est_lat, est_lon):
        """Calculate geodesic distance between true and estimated positions in meters"""
        distance = geodesic((true_lat, true_lon), (est_lat, est_lon)).meters
        return distance

    def process_data(self):
        """Process raw tracking data into structured format for analysis"""
        print(f"+ Processing tracking data")

        processed_records = []

        for file_key, variance_blocks in self.tracking_data.items():
            # variance_blocks is an array like:
            # [{"data": [...], "variance": 1e-05}, {"data": [...], "variance": 0.0001}, ...]

            for variance_block in variance_blocks:
                variance = variance_block["variance"]
                snapshots = variance_block["data"]

                for snapshot in snapshots:
                    # Extract basic info
                    record = {
                        "file": file_key,
                        "variance": variance,
                        "time_spent": snapshot["time_spent"],
                        "true_lat": snapshot["tracking"]["truth_pos"][0],
                        "true_lon": snapshot["tracking"]["truth_pos"][1],
                    }

                    # Extract algorithm results and calculate errors
                    for algo in self.algorithm_names:
                        if (
                            algo in snapshot["tracking"]
                            and snapshot["tracking"][algo] is not None
                        ):
                            est_lat, est_lon = snapshot["tracking"][algo]
                            record[f"{algo}_lat"] = est_lat
                            record[f"{algo}_lon"] = est_lon

                            # Calculate geodesic distance error in meters
                            error_meters = self._calculate_distance_error_meters(
                                record["true_lat"], record["true_lon"], est_lat, est_lon
                            )
                            record[f"{algo}_error_meters"] = error_meters

                            # Also keep the degree-based error for compatibility
                            error_degrees = np.sqrt(
                                (est_lat - record["true_lat"]) ** 2
                                + (est_lon - record["true_lon"]) ** 2
                            )
                            record[f"{algo}_error_degrees"] = error_degrees

                            # Calculate individual lat/lon errors in degrees and meters
                            record[f"{algo}_lat_error_degrees"] = abs(
                                est_lat - record["true_lat"]
                            )
                            record[f"{algo}_lon_error_degrees"] = abs(
                                est_lon - record["true_lon"]
                            )

                            # Calculate lat/lon errors in meters (approximate)
                            lat_error_meters = self._calculate_distance_error_meters(
                                record["true_lat"],
                                record["true_lon"],
                                est_lat,
                                record["true_lon"],
                            )
                            lon_error_meters = self._calculate_distance_error_meters(
                                record["true_lat"],
                                record["true_lon"],
                                record["true_lat"],
                                est_lon,
                            )
                            record[f"{algo}_lat_error_meters"] = lat_error_meters
                            record[f"{algo}_lon_error_meters"] = lon_error_meters

                        else:
                            # Algorithm failed
                            record[f"{algo}_lat"] = None
                            record[f"{algo}_lon"] = None
                            record[f"{algo}_error_meters"] = None
                            record[f"{algo}_error_degrees"] = None
                            record[f"{algo}_lat_error_degrees"] = None
                            record[f"{algo}_lon_error_degrees"] = None
                            record[f"{algo}_lat_error_meters"] = None
                            record[f"{algo}_lon_error_meters"] = None

                    processed_records.append(record)

        self.processed_data = pd.DataFrame(processed_records)
        print(f"| Processed {len(processed_records)} records")
        print(f"+ Data processing completed")

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for all algorithms"""
        print(f"+ Calculating performance statistics")

        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")

        stats = {}

        # Overall statistics
        stats["total_records"] = len(self.processed_data)
        stats["unique_variances"] = sorted(self.processed_data["variance"].unique())
        stats["time_range"] = [
            self.processed_data["time_spent"].min(),
            self.processed_data["time_spent"].max(),
        ]

        # Per-algorithm statistics
        stats["algorithms"] = {}

        for algo in self.algorithm_names:
            error_col = f"{algo}_error_meters"
            if error_col in self.processed_data.columns:
                valid_data = self.processed_data[self.processed_data[error_col].notna()]

                if len(valid_data) > 0:
                    stats["algorithms"][algo] = {
                        "success_rate": len(valid_data) / len(self.processed_data),
                        "mean_error_meters": valid_data[error_col].mean(),
                        "std_error_meters": valid_data[error_col].std(),
                        "median_error_meters": valid_data[error_col].median(),
                        "min_error_meters": valid_data[error_col].min(),
                        "max_error_meters": valid_data[error_col].max(),
                        "q25_error_meters": valid_data[error_col].quantile(0.25),
                        "q75_error_meters": valid_data[error_col].quantile(0.75),
                    }
                else:
                    stats["algorithms"][algo] = {"success_rate": 0.0}

        # Per-variance statistics
        stats["by_variance"] = {}
        for variance in stats["unique_variances"]:
            variance_data = self.processed_data[
                self.processed_data["variance"] == variance
            ]
            stats["by_variance"][variance] = {}

            for algo in self.algorithm_names:
                error_col = f"{algo}_error_meters"  # Use meters for variance analysis
                if error_col in variance_data.columns:
                    valid_data = variance_data[variance_data[error_col].notna()]

                    if len(valid_data) > 0:
                        stats["by_variance"][variance][algo] = {
                            "mean_error_meters": valid_data[error_col].mean(),
                            "std_error_meters": valid_data[error_col].std(),
                            "success_rate": len(valid_data) / len(variance_data),
                        }
        print(f"+ Statistics calculation completed")
        return stats

    def generate_report(self, save_report: bool = True) -> str:
        """Generate comprehensive analysis report"""
        print(f"+ Generating analysis report")

        stats = self.calculate_statistics()

        report_lines = []
        report_lines.append("# Tracking Algorithm Analysis Report")
        report_lines.append(f"Generated from: {self.results_folder}")
        report_lines.append("")

        # Overview
        report_lines.append("## Overview")
        report_lines.append(f"- Total records analyzed: {stats['total_records']}")
        report_lines.append(f"- Variance levels tested: {stats['unique_variances']}")
        report_lines.append(
            f"- Time range: {stats['time_range'][0]}s to {stats['time_range'][1]}s"
        )
        report_lines.append("")

        # Algorithm performance summary
        report_lines.append("## Algorithm Performance Summary")
        report_lines.append(
            "| Algorithm | Success Rate | Mean Error (m) | Std Error (m) | Median Error (m) |"
        )
        report_lines.append(
            "|-----------|--------------|----------------|---------------|------------------|"
        )

        for algo in self.algorithm_names:
            if algo in stats["algorithms"]:
                algo_stats = stats["algorithms"][algo]
                if "mean_error_meters" in algo_stats:
                    report_lines.append(
                        f"| {algo.upper()} | {algo_stats['success_rate']:.3f} | "
                        f"{algo_stats['mean_error_meters']:.2f} | {algo_stats['std_error_meters']:.2f} | "
                        f"{algo_stats['median_error_meters']:.2f} |"
                    )
                else:
                    report_lines.append(
                        f"| {algo.upper()} | {algo_stats['success_rate']:.3f} | - | - | - |"
                    )

        report_lines.append("")

        # Performance by variance
        report_lines.append("## Performance by Variance Level")
        for variance in stats["unique_variances"]:
            report_lines.append(f"### Variance: {variance}")
            report_lines.append(
                "| Algorithm | Mean Error (m) | Std Error (m) | Success Rate |"
            )
            report_lines.append(
                "|-----------|----------------|---------------|--------------|"
            )

            if variance in stats["by_variance"]:
                for algo in self.algorithm_names:
                    if algo in stats["by_variance"][variance]:
                        algo_stats = stats["by_variance"][variance][algo]
                        report_lines.append(
                            f"| {algo.upper()} | {algo_stats['mean_error_meters']:.2f} | "
                            f"{algo_stats['std_error_meters']:.2f} | {algo_stats['success_rate']:.3f} |"
                        )
            report_lines.append("")

        report_content = "\n".join(report_lines)

        if save_report:
            report_file = os.path.join(self.results_folder, "analysis_report.md")
            with open(report_file, "w") as f:
                f.write(report_content)
            print(f"| Report saved to: {report_file}")

        print(f"+ Report generation completed")
        return report_content

    def plot_performance_vs_variance(self, save_plots: bool = True):
        """Plot algorithm performance vs noise variance - separate plots"""
        print(f"+ Generating performance vs variance plots")

        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")

        # Set up the plotting style
        plt.style.use("default")
        colors = {"tdoa": "red", "tmm": "black", "sr_ls": "blue"}
        variances = sorted(self.processed_data["variance"].unique())

        # Plot 1: MSE vs Variance (Log-Log)
        plt.figure(figsize=(10, 8))
        for algo in self.algorithm_names:
            if algo == "centroid":
                continue

            mse_values = []
            for variance in variances:
                variance_data = self.processed_data[
                    self.processed_data["variance"] == variance
                ]
                error_col = f"{algo}_error_meters"
                valid_data = variance_data[variance_data[error_col].notna()]

                if len(valid_data) > 0:
                    mse = (valid_data[error_col] ** 2).mean()
                    mse_values.append(mse)
                else:
                    mse_values.append(np.nan)

            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(mse_values)
            if np.any(valid_indices):
                valid_variances = [
                    variances[i] for i in range(len(variances)) if valid_indices[i]
                ]
                valid_mse = [
                    mse_values[i] for i in range(len(mse_values)) if valid_indices[i]
                ]

                plt.loglog(
                    valid_variances,
                    valid_mse,
                    "o-",
                    label=algo.upper(),
                    color=colors.get(algo, "gray"),
                    linewidth=2,
                    markersize=8,
                )

        plt.xlabel("TOA Variance", fontsize=12)
        plt.ylabel("MSE (m²)", fontsize=12)
        plt.title("Mean Square Error vs TOA Variance", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if save_plots:
            plot_file = os.path.join(self.results_folder, "mse_vs_variance.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"| MSE vs Variance plot saved to: {plot_file}")
        # plt.show()

        # # Plot 2: Bar chart comparison
        plt.figure(figsize=(12, 8))
        bar_width = 0.25
        x = np.arange(len(variances))

        for i, algo in enumerate(self.algorithm_names):
            if algo == "centroid":
                continue

            mse_values = []
            for variance in variances:
                variance_data = self.processed_data[
                    self.processed_data["variance"] == variance
                ]
                error_col = f"{algo}_error_meters"
                valid_data = variance_data[variance_data[error_col].notna()]

                if len(valid_data) > 0:
                    mse = (valid_data[error_col] ** 2).mean()
                    mse_values.append(mse)
                else:
                    mse_values.append(0)

            plt.bar(
                x + i * bar_width,
                mse_values,
                bar_width,
                label=algo.upper(),
                color=colors.get(algo, "gray"),
                alpha=0.8,
            )

        plt.xlabel("TOA Variance", fontsize=12)
        plt.ylabel("MSE (m²)", fontsize=12)
        plt.title(
            "MSE Comparison by Algorithm and Variance Level",
            fontsize=14,
            fontweight="bold",
        )
        plt.yscale("log")
        plt.xticks(x + bar_width, [f"{v:.0e}" for v in variances], rotation=45)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plots:
            plot_file = os.path.join(self.results_folder, "mse_bar_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"| MSE Bar Comparison plot saved to: {plot_file}")
        # plt.show()

        # Plot 3: Mean Error vs Variance
        plt.figure(figsize=(10, 8))
        for algo in self.algorithm_names:
            if algo == "centroid":
                continue

            mean_errors = []
            for variance in variances:
                variance_data = self.processed_data[
                    self.processed_data["variance"] == variance
                ]
                error_col = f"{algo}_error_meters"
                valid_data = variance_data[variance_data[error_col].notna()]

                if len(valid_data) > 0:
                    mean_error = valid_data[error_col].mean()
                    mean_errors.append(mean_error)
                else:
                    mean_errors.append(np.nan)

            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(mean_errors)
            if np.any(valid_indices):
                valid_variances = [
                    variances[i] for i in range(len(variances)) if valid_indices[i]
                ]
                valid_errors = [
                    mean_errors[i] for i in range(len(mean_errors)) if valid_indices[i]
                ]

                plt.loglog(
                    valid_variances,
                    valid_errors,
                    "o-",
                    label=algo.upper(),
                    color=colors.get(algo, "gray"),
                    linewidth=2,
                    markersize=8,
                )

        plt.xlabel("TOA Variance", fontsize=12)
        plt.ylabel("Mean Error (m)", fontsize=12)
        plt.title("Mean Error vs TOA Variance", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        if save_plots:
            plot_file = os.path.join(self.results_folder, "mean_error_vs_variance.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"| Mean Error vs Variance plot saved to: {plot_file}")
        # plt.show()

        print(f"+ Performance vs variance plots completed\n")

    def plot_error_distributions(self, save_plots: bool = True):
        """Plot error distributions for each algorithm - separate plots for each variance"""
        print(f"+ Generating error distribution plots")

        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")

        variances = sorted(self.processed_data["variance"].unique())
        colors = {"tdoa": "red", "tmm": "black", "sr_ls": "blue"}

        for variance in variances:
            plt.figure(figsize=(10, 8))

            variance_data = self.processed_data[
                self.processed_data["variance"] == variance
            ]

            # Collect error data for all algorithms
            error_data = []
            labels = []

            for algo in self.algorithm_names:
                error_col = f"{algo}_error_meters"
                if error_col in variance_data.columns:
                    valid_errors = variance_data[variance_data[error_col].notna()][
                        error_col
                    ]
                    if len(valid_errors) > 0:
                        error_data.append(valid_errors.values)
                        labels.append(algo.upper())

            if error_data:
                # Create box plot
                bp = plt.boxplot(
                    error_data,
                    labels=labels,
                    patch_artist=True,
                    showfliers=True,
                    whis=1.5,
                )

                # Color the boxes
                for patch, algo in zip(bp["boxes"], [l.lower() for l in labels]):
                    patch.set_facecolor(colors.get(algo, "gray"))
                    patch.set_alpha(0.7)

                plt.ylabel("Error (meters)", fontsize=12)
                plt.title(
                    f"Error Distribution - Variance: {variance:.0e}",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.grid(True, alpha=0.3)
                plt.yscale("log")
                plt.tight_layout()

                if save_plots:
                    plot_file = os.path.join(
                        self.results_folder,
                        f"error_distribution_var_{variance:.0e}.png",
                    )
                    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                    print(f"| Error distribution plot saved to: {plot_file}")

                # plt.show()
            else:
                print(f"| No valid data for variance {variance}")

        print(f"+ Error distribution plots completed\n")

    def plot_spatial_analysis(self, save_plots: bool = True):
        """Plot spatial distribution of estimates vs truth - individual plots for each algorithm"""
        print(f"+ Generating spatial analysis plots")

        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")

        # Use the lowest variance for clearest spatial visualization
        # min_variance = self.processed_data["variance"].min()
        # spatial_data = self.processed_data[
        #     self.processed_data["variance"] == min_variance
        # ]

        colors = {"tdoa": "red", "tmm": "black", "sr_ls": "blue"}

        # Individual plots for each algorithm
        for variance in np.unique(self.processed_data["variance"]):
            spatial_data = self.processed_data[
                self.processed_data["variance"] == variance
            ]

            for algo in self.algorithm_names:
                if algo == "centroid":
                    continue

                plt.figure(figsize=(10, 10))

                # Get algorithm estimates
                lat_col = f"{algo}_lat"
                lon_col = f"{algo}_lon"

                if lat_col in spatial_data.columns and lon_col in spatial_data.columns:
                    valid_data = spatial_data[
                        spatial_data[lat_col].notna() & spatial_data[lon_col].notna()
                    ]

                    if len(valid_data) > 0:
                        # Plot true positions
                        plt.scatter(
                            valid_data["true_lon"],
                            valid_data["true_lat"],
                            c="black",
                            marker="*",
                            s=100,
                            label="True Position",
                            alpha=0.8,
                            edgecolors="white",
                            linewidth=1,
                        )

                        # Plot estimated positions
                        plt.scatter(
                            valid_data[lon_col],
                            valid_data[lat_col],
                            c=colors.get(algo, "gray"),
                            marker="o",
                            s=50,
                            label=f"{algo.upper()} Estimate",
                            alpha=0.6,
                        )

                        # Draw lines connecting true and estimated positions (for first few points)
                        for idx, (_, row) in enumerate(valid_data.head(20).iterrows()):
                            plt.plot(
                                [row["true_lon"], row[lon_col]],
                                [row["true_lat"], row[lat_col]],
                                "k--",
                                alpha=0.3,
                                linewidth=0.5,
                            )

                        plt.xlabel("Longitude", fontsize=12)
                        plt.ylabel("Latitude", fontsize=12)
                        plt.title(
                            f"{algo.upper()} Spatial Distribution (Variance: {variance:.0e})",
                            fontsize=14,
                            fontweight="bold",
                        )
                        plt.grid(True, alpha=0.3)
                        plt.legend(fontsize=12)

                        # Set equal aspect ratio
                        plt.axis("equal")
                        plt.tight_layout()

                        if save_plots:
                            plot_file = os.path.join(
                                self.results_folder,
                                f"spatial_analysis_{algo}_{variance:.0e}.png",
                            )
                            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                            print(f"| Spatial analysis plot saved to: {plot_file}")

                    else:
                        print(f"| No valid data for {algo.upper()}")
                else:
                    print(f"| No data columns for {algo.upper()}")

        print(f"+ Spatial analysis plots completed\n")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        # Load and process data
        self.load_tracking_data()
        self.process_data()

        # print(self.calculate_statistics())

        # # Generate all outputs
        self.generate_report()
        self.plot_performance_vs_variance()
        self.plot_error_distributions()
        self.plot_spatial_analysis()


def parse_args():
    """Parse command line arguments for analyzer"""
    if len(sys.argv) < 2:
        print("Error: You must specify the output folder path.")
        print("Usage: python analyzer.py /path/to/output/folder")
        print("Example: python analyzer.py output/sample_simulation")
        sys.exit(1)

    output_folder = sys.argv[1]
    return output_folder


def main():
    """Main function for analysis module"""
    try:
        output_folder = parse_args()

        print(f"+ Starting Darkships Tracking Analysis")
        print(f"| Output folder: {output_folder}")

        # Check if output folder exists
        if not os.path.exists(output_folder):
            print(f"| Error: Output folder not found: {output_folder}")
            sys.exit(1)

        # Check if tracking folder exists
        tracking_folder = f"{output_folder}/tracking"
        if not os.path.exists(tracking_folder):
            print(f"| Error: Tracking folder not found: {tracking_folder}")
            sys.exit(1)

        # Create and run analyzer
        analyzer = TrackingAnalyzer(output_folder)
        analyzer.run_full_analysis()

        print("+ Analysis completed successfully!")

    except KeyboardInterrupt:
        print("\n+ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"| Error: Analysis failed - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
