import sys

from simulation import Simulation
from tracking import Tracking


def main():
    config_path, run_tracking = parse_args()

    if run_tracking:
        tracker = Tracking(config_path)
        tracker.run()
    else:
        sim = Simulation(config_path, 60)
        sim.run(5)


def parse_args():
    """Parse arguments and return (config_path, run_tracking_flag)"""
    if len(sys.argv) < 2:
        print("Error: You must specify the path of the config file.")
        print("Usage: python main.py /path/to/file.yaml [-t]")
        sys.exit(1)

    config_path = sys.argv[1]
    run_tracking = "-t" in sys.argv
    return config_path, run_tracking


if __name__ == "__main__":
    main()
