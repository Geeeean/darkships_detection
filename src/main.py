import sys
import threading

from simulation import Simulation
import multiprocessing


def main():
    config_path = get_config_path()
    sim = Simulation(config_path)

    sim.initialize_environment()
    sim.start()

    p = multiprocessing.Process(target=sim.estimate_ds_positions)
    p.start()

    # sim.plot()


def get_config_path():
    if len(sys.argv) < 2:
        print("Error: You must specify the path of the config file.")
        print("Usage: python main.py /path/to/file.yaml")
        sys.exit(1)

    return sys.argv[1]


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
