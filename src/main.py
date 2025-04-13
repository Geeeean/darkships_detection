import sys

from server import Server
from simulation import Simulation
import multiprocessing


def main():
    config_path = get_config_path()

    sim_server_queue = multiprocessing.Queue()
    server_sim_queue = multiprocessing.Queue()

    sim_process = multiprocessing.Process(
        target=run_simulation,
        args=(config_path, sim_server_queue, server_sim_queue, 1000, 1000),
    )
    sim_process.start()

    server = Server(sim_server_queue, server_sim_queue)
    server.run()

    sim_process.join()


def run_simulation(
    config_path, sim_server_queue, server_sim_queue, total_steps, delta_t_sec
):
    sim = Simulation(config_path, server_sim_queue, sim_server_queue)
    sim.initialize_environment()

    sim.run(total_steps, delta_t_sec)


def get_config_path():
    if len(sys.argv) < 2:
        print("Error: You must specify the path of the config file.")
        print("Usage: python main.py /path/to/file.yaml")
        sys.exit(1)

    return sys.argv[1]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
