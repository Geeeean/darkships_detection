import sys
from simulation import SimulationManager


def main():
    config_path = get_config_path()
    sim_manager = SimulationManager(config_path);

    sim_manager.initialize_environment()
    sim_manager.plot_simulation()

def get_config_path():
    if len(sys.argv) < 2:
        print("Error: You must specify the path of the config file.")
        print("Usage: python main.py /path/to/file.yaml")
        sys.exit(1)

    return sys.argv[1]


if __name__ == "__main__":
    main()
