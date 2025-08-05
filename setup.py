import argparse
import json
from random import shuffle
from src.setup import SETUP_FUNCS
from src.util import REPO_DIR


def setup(args):
    assert args.experiment in SETUP_FUNCS, "The selected experiment does not exist."
    experiments = SETUP_FUNCS[args.experiment]()
    experiments = list(map(lambda x: x.as_dict(), experiments))
    if not args.stable:
        shuffle(experiments)
    with open(args.o, "w", newline="") as f:
        json.dump(experiments, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        prog="Experiment setup",
        description="Setup an experiment by filling a json with run descriptions",
    )
    parser.add_argument("experiment", choices=SETUP_FUNCS.keys())
    parser.add_argument("-o", default=str(REPO_DIR / "experiments.json"))
    parser.add_argument("--stable", action="store_true", help="Experiments are shuffled by default for load balancing, set this flag to disable shuffling.")

    args = parser.parse_args()
    setup(args)


if __name__ == "__main__":
    main()
