import sys

from pandas import pandas
from simulation import Simulation
from arlpy import uwapm as pm
import numpy as np


def main():
    config_path = get_config_path()
    sim = Simulation(config_path)

    sim.initialize_environment()
    sim.start()
    sim.plot()

def get_config_path():
    if len(sys.argv) < 2:
        print("Error: You must specify the path of the config file.")
        print("Usage: python main.py /path/to/file.yaml")
        sys.exit(1)

    return sys.argv[1]


if __name__ == "__main__":
    main()
