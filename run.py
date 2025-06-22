#!/usr/bin/env python3
"""
Simple script to run the complete pipeline: simulation -> tracking -> analysis
Usage: python run.py config/config1.yaml [iterations]
"""

import sys
import subprocess
import yaml
import os


def load_config(config_path):
    """Load YAML config and extract output path"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        output_path = os.path.join(config['output_path'], config['name'])
        return output_path
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)


def run_command(cmd):
    """Run a shell command and handle errors"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py config/file.yaml [iterations]")
        print("Example: python run.py config/config1.yaml 100")
        sys.exit(1)

    config_file = sys.argv[1]
    iterations = sys.argv[2] if len(sys.argv) > 2 else "100"

    # Extract output path from config
    output_path = load_config(config_file)

    print("=== Pipeline Configuration ===")
    print(f"Config: {config_file}")
    print(f"Iterations: {iterations}")
    print(f"Output: {output_path}")
    print()

    # Run simulation
    print("=== Running Simulation ===")
    if not run_command(["python3", "src/simulation.py", config_file, "-i", iterations]):
        print("Simulation failed!")
        sys.exit(1)

    print()

    # Run tracking
    print("=== Running Tracking ===")
    if not run_command(["python3", "src/tracking.py", output_path]):
        print("Tracking failed!")
        sys.exit(1)

    print()

    # Run analysis
    print("=== Running Analysis ===")
    if not run_command(["python3", "src/analyzer.py", output_path]):
        print("Analysis failed!")
        sys.exit(1)

    print()
    print("=== Pipeline Complete ===")
    print(f"Results available in: {output_path}")


if __name__ == "__main__":
    main()
