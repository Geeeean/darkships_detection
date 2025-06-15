import sys
from simulation import Simulation
from tracking import Tracking


def main():
    command, path, iterations = parse_args()

    if command == "tracking":
        tracker = Tracking(path)  # path è la cartella di output
        tracker.run()
    else:  # simulation
        sim = Simulation(path, 60, iterations)  # path è il config file
        sim.run(5)


def parse_args():
    """Parse arguments and return (command, path, iterations)"""
    if len(sys.argv) < 2:
        print("Error: You must specify a path.")
        print("Usage:")
        print("  Simulation: python main.py /path/to/config.yaml [-i <iterations>]")
        print("  Tracking:   python main.py /path/to/output/folder -t")
        sys.exit(1)

    path = sys.argv[1]
    run_tracking = "-t" in sys.argv
    iterations = 1

    if run_tracking:
        if len(sys.argv) > 3:  # path + -t = 3 args totali
            print("Error: When using -t, no other options are allowed.")
            print("Usage: python main.py /path/to/output/folder -t")
            sys.exit(1)
        return "tracking", path, iterations

    # Per simulation, controlla se c'è -i
    if "-i" in sys.argv:
        try:
            i_index = sys.argv.index("-i")
            if i_index + 1 >= len(sys.argv):
                print("Error: -i option requires a number.")
                print("Usage: python main.py /path/to/config.yaml [-i <iterations>]")
                sys.exit(1)

            iterations = int(sys.argv[i_index + 1])
            if iterations <= 0:
                print("Error: iterations must be a positive number.")
                sys.exit(1)

        except ValueError:
            print("Error: iterations must be a valid integer.")
            print("Usage: python main.py /path/to/config.yaml [-i <iterations>]")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing -i option: {e}")
            sys.exit(1)

    return "simulation", path, iterations


if __name__ == "__main__":
    main()
