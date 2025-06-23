import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import yaml
from typing import Dict, Any
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
        self.algorithm_names = ["tdoa", "tmm", "sr_ls"]

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
                            try:
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
                            except:
                                continue

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
            "| Algorithm | Mean Error (m) | Std Error (m) | Median Error (m) |"
        )
        report_lines.append(
            "|-----------|----------------|---------------|------------------|"
        )

        for algo in self.algorithm_names:
            if algo in stats["algorithms"]:
                algo_stats = stats["algorithms"][algo]
                if "mean_error_meters" in algo_stats:
                    report_lines.append(
                        f"| {algo.upper()} | "
                        f"{algo_stats['mean_error_meters']:.2f} | {algo_stats['std_error_meters']:.2f} | "
                        f"{algo_stats['median_error_meters']:.2f} |"
                    )
                # else:
                #     report_lines.append(
                #         f"| {algo.upper()} | {algo_stats['success_rate']:.3f} | - | - | - |"
                #     )

        report_lines.append("")

        # Performance by variance
        report_lines.append("## Performance by Variance Level")
        for variance in stats["unique_variances"]:
            report_lines.append(f"### Variance: {variance}")
            report_lines.append(
                "| Algorithm | Mean Error (m) | Std Error (m) |"
            )
            report_lines.append(
                "|-----------|----------------|---------------|"
            )

            if variance in stats["by_variance"]:
                for algo in self.algorithm_names:
                    if algo in stats["by_variance"][variance]:
                        algo_stats = stats["by_variance"][variance][algo]
                        report_lines.append(
                            f"| {algo.upper()} | {algo_stats['mean_error_meters']:.2f} | "
                            f"{algo_stats['std_error_meters']:.2f} |"
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

    def plot_environment_map(self, variance_to_plot=None, save_plots: bool = True):
        """
        Plot environment map showing hydrophones and ship movement

        Args:
            variance_to_plot: Specific variance level to plot (if None, uses lowest variance)
            save_plots: Whether to save the plot
        """
        print(f"+ Generating environment map")

        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")

        # Select variance level to plot
        if variance_to_plot is None:
            variance_to_plot = self.processed_data["variance"].min()

        # Filter data for selected variance
        map_data = self.processed_data[
            self.processed_data["variance"] == variance_to_plot
        ].sort_values("time_spent")

        if len(map_data) == 0:
            print(f"| No data found for variance {variance_to_plot}")
            return

        # Extract hydrophone positions from config
        hydrophones = self.config["hydrophones_config"]["hydrophones"]
        hydro_lats = [h["coordinates"][0] for h in hydrophones]
        hydro_lons = [h["coordinates"][1] for h in hydrophones]

        # Extract ship trajectory
        ship_lats = map_data["true_lat"].values
        ship_lons = map_data["true_lon"].values
        times = map_data["time_spent"].values

        # Create the plot
        plt.figure(figsize=(14, 10))

        # Plot hydrophones
        plt.scatter(
            hydro_lons,
            hydro_lats,
            c="blue",
            marker="^",
            s=300,
            label="Idrofoni",
            edgecolors="white",
            linewidth=2,
            zorder=5,
        )

        # Add hydrophone labels
        for i, (lon, lat) in enumerate(zip(hydro_lons, hydro_lats)):
            plt.annotate(
                f"H{i+1}",
                (lon, lat),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                color="blue",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Plot ship trajectory
        plt.plot(
            ship_lons,
            ship_lats,
            "k-",
            linewidth=3,
            label="Traiettoria nave",
            alpha=0.8,
            zorder=3,
        )

        # Plot ship positions with time gradient
        scatter = plt.scatter(
            ship_lons,
            ship_lats,
            c=times,
            cmap="Reds",
            s=80,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.5,
            zorder=4,
        )

        # Mark start and end positions
        plt.scatter(
            ship_lons[0],
            ship_lats[0],
            c="green",
            marker="*",
            s=400,
            label="Posizione iniziale",
            edgecolors="black",
            linewidth=2,
            zorder=6,
        )

        plt.scatter(
            ship_lons[-1],
            ship_lats[-1],
            c="red",
            marker="*",
            s=400,
            label="Posizione finale",
            edgecolors="black",
            linewidth=2,
            zorder=6,
        )

        # Calculate tight bounds based on actual data
        all_lats = hydro_lats + ship_lats.tolist()
        all_lons = hydro_lons + ship_lons.tolist()

        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)

        # Calculate ranges
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        # If range is very small, set minimum range
        min_range = 0.005  # About 500m at this latitude
        if lat_range < min_range:
            lat_center = (lat_min + lat_max) / 2
            lat_min = lat_center - min_range / 2
            lat_max = lat_center + min_range / 2
            lat_range = min_range

        if lon_range < min_range:
            lon_center = (lon_min + lon_max) / 2
            lon_min = lon_center - min_range / 2
            lon_max = lon_center + min_range / 2
            lon_range = min_range

        # Add margin (30% for better visibility)
        margin = 0.3
        lat_margin = lat_range * margin
        lon_margin = lon_range * margin

        plt.xlim(lon_min - lon_margin, lon_max + lon_margin)
        plt.ylim(lat_min - lat_margin, lat_max + lat_margin)

        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=plt.gca(), shrink=0.8)
        cbar.set_label("Tempo (s)", fontsize=12, fontweight="bold")

        # Labels and title
        plt.xlabel("Longitudine (°)", fontsize=14, fontweight="bold")
        plt.ylabel("Latitudine (°)", fontsize=14, fontweight="bold")
        plt.title(
            f"Mappa Ambiente - Idrofoni e Movimento Nave",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Grid and legend
        plt.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
        plt.legend(loc="best", fontsize=12, framealpha=0.9)

        # Set tick format to show more decimal places
        plt.gca().ticklabel_format(useOffset=False, style="plain")

        # Add scale information
        # Calculate approximate scale in meters
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2

        # Calculate distance for the plot range
        total_distance = 0
        if len(ship_lats) > 1:
            for i in range(1, len(ship_lats)):
                dist = geodesic(
                    (ship_lats[i - 1], ship_lons[i - 1]), (ship_lats[i], ship_lons[i])
                ).meters
                total_distance += dist

        # Add text box with info
        info_text = f"Area: {lat_range:.4f}° × {lon_range:.4f}°\n"
        if total_distance > 0:
            info_text += f"Distanza nave: {total_distance:.0f} m\n"
            info_text += f"Durata: {times[-1] - times[0]:.0f} s"

        plt.text(
            0.02,
            0.98,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        if save_plots:
            plot_file = os.path.join(
                self.results_folder, f"environment_map_var_{variance_to_plot:.0e}.png"
            )
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"| Environment map saved to: {plot_file}")

        # Print statistics
        print(
            f"| Map bounds: Lat [{lat_min:.5f}, {lat_max:.5f}], Lon [{lon_min:.5f}, {lon_max:.5f}]"
        )
        if total_distance > 0:
            print(
                f"| Ship trajectory: {total_distance:.1f} m in {times[-1] - times[0]:.0f} s"
            )
            print(
                f"| Average speed: {total_distance / (times[-1] - times[0]) * 3.6:.1f} km/h"
            )

        print(f"+ Environment map completed\n")

    def plot_algorithm_comparison_map(
        self, variance_to_plot=None, save_plots: bool = True
    ):
        """
        Plot map comparing algorithm estimates with true positions - separate files
        """
        print(f"+ Generating algorithm comparison maps")
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")

        # Select variance level to plot
        if variance_to_plot is None:
            variance_to_plot = self.processed_data["variance"].min()

        # Filter data for selected variance
        map_data = self.processed_data[
            self.processed_data["variance"] == variance_to_plot
        ].sort_values("time_spent")

        if len(map_data) == 0:
            print(f"| No data found for variance {variance_to_plot}")
            return

        # Extract hydrophone positions
        hydrophones = self.config["hydrophones_config"]["hydrophones"]
        hydro_lats = [h["coordinates"][0] for h in hydrophones]
        hydro_lons = [h["coordinates"][1] for h in hydrophones]

        # Colors for algorithms
        colors = {"tdoa": "red", "tmm": "black", "sr_ls": "blue"}

        algorithms = ["tdoa", "tmm", "sr_ls"]

        # Calculate bounds once for all plots (for consistency)
        all_lats = hydro_lats + map_data["true_lat"].tolist()
        all_lons = hydro_lons + map_data["true_lon"].tolist()

        # Add algorithm estimates to bounds calculation
        for algo in algorithms:
            algo_data = map_data[map_data[f"{algo}_lat"].notna()]
            if len(algo_data) > 0:
                all_lats.extend(algo_data[f"{algo}_lat"].tolist())
                all_lons.extend(algo_data[f"{algo}_lon"].tolist())

        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)

        # Calculate ranges with minimum range
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        min_range = 0.005  # About 500m at this latitude

        if lat_range < min_range:
            lat_center = (lat_min + lat_max) / 2
            lat_min = lat_center - min_range / 2
            lat_max = lat_center + min_range / 2
            lat_range = min_range

        if lon_range < min_range:
            lon_center = (lon_min + lon_max) / 2
            lon_min = lon_center - min_range / 2
            lon_max = lon_center + min_range / 2
            lon_range = min_range

        # Add margin
        margin = 0.2
        lat_margin = lat_range * margin
        lon_margin = lon_range * margin

        # Create separate plot for each algorithm
        for algo in algorithms:
            plt.figure(figsize=(12, 10))

            # Plot hydrophones
            plt.scatter(
                hydro_lons,
                hydro_lats,
                c="blue",
                marker="^",
                s=200,
                label="Idrofoni",
                edgecolors="white",
                linewidth=2,
                zorder=5,
            )

            # Add hydrophone labels
            for i, (lon, lat) in enumerate(zip(hydro_lons, hydro_lats)):
                plt.annotate(
                    f"H{i+1}",
                    (lon, lat),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color="blue",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

            # Plot true trajectory
            plt.plot(
                map_data["true_lon"],
                map_data["true_lat"],
                "k-",
                linewidth=3,
                label="Traiettoria reale",
                alpha=0.8,
                zorder=3,
            )

            # Plot algorithm estimates
            algo_data = map_data[map_data[f"{algo}_lat"].notna()]
            if len(algo_data) > 0:
                plt.plot(
                    algo_data[f"{algo}_lon"],
                    algo_data[f"{algo}_lat"],
                    "o-",
                    color=colors[algo],
                    linewidth=2,
                    markersize=6,
                    label=f"Stima {algo.upper()}",
                    alpha=0.9,
                    zorder=4,
                )

                # Connect true and estimated positions with lines (show first 10 for clarity)
                for _, row in algo_data.head(10).iterrows():
                    plt.plot(
                        [row["true_lon"], row[f"{algo}_lon"]],
                        [row["true_lat"], row[f"{algo}_lat"]],
                        "gray",
                        linestyle=":",
                        alpha=0.4,
                        linewidth=1,
                    )

            # Mark start and end positions
            plt.scatter(
                map_data["true_lon"].iloc[0],
                map_data["true_lat"].iloc[0],
                c="green",
                marker="*",
                s=300,
                label="Posizione iniziale",
                edgecolors="black",
                linewidth=2,
                zorder=6,
            )

            plt.scatter(
                map_data["true_lon"].iloc[-1],
                map_data["true_lat"].iloc[-1],
                c="red",
                marker="*",
                s=300,
                label="Posizione finale",
                edgecolors="black",
                linewidth=2,
                zorder=6,
            )

            # Set consistent bounds for all plots
            plt.xlim(lon_min - lon_margin, lon_max + lon_margin)
            plt.ylim(lat_min - lat_margin, lat_max + lat_margin)

            # Labels and title
            plt.xlabel("Longitudine (°)", fontsize=12, fontweight="bold")
            plt.ylabel("Latitudine (°)", fontsize=12, fontweight="bold")
            plt.title(
                f"Confronto Localizzazione: {algo.upper()}\n(TOA Variance: {variance_to_plot:.0e})",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )

            # Grid and legend
            plt.grid(True, alpha=0.4)
            plt.legend(fontsize=10, framealpha=0.9)

            # Format ticks
            plt.gca().ticklabel_format(useOffset=False, style="plain")

            # Calculate and display error statistics for this algorithm
            if len(algo_data) > 0:
                error_stats = algo_data[f"{algo}_error_meters"]
                mean_error = error_stats.mean()
                std_error = error_stats.std()

                stats_text = f"Errore medio: {mean_error:.1f} m\nStd: {std_error:.1f} m\nPunti: {len(algo_data)}"
                plt.text(
                    0.02,
                    0.02,
                    stats_text,
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8
                    ),
                )

            plt.tight_layout()

            if save_plots:
                plot_file = os.path.join(
                    self.results_folder,
                    f"algorithm_map_{algo}_var_{variance_to_plot:.0e}.png",
                )
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                print(f"| {algo.upper()} map saved to: {plot_file}")

            # Close the figure to free memory
            plt.close()

        print(f"+ Algorithm comparison maps completed\n")

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

        self.plot_environment_map()
        self.plot_algorithm_comparison_map()


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
