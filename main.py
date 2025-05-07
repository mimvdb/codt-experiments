import argparse
import datetime
import json
import numpy as np
import sys
from src.methods import Run, RunParams, get_classification_method, get_regression_method
from src.setup.debug import setup_debug
from src.util import REPO_DIR

SETUP_FUNCS = {
    "debug": setup_debug
}

# Allow numpy primitives to encode to JSON
def encode_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()


def run(args):
    experiments = json.load(sys.stdin)
    runs = []
    for e in experiments:
        params = RunParams(**e)
        if params.task == "classification":
            method = get_classification_method(params.method)
        elif params.task == "regression":
            method = get_regression_method(params.method)
        else:
            assert False, "Only classification and regression is supported"
        output = method.run(params)
        runs.append(Run(params, output).as_dict())
    with open(args.o, "w", newline="") as f:
        json.dump(runs, f, indent=4, default=encode_numpy)


def setup(args):
    assert args.experiment in SETUP_FUNCS, "The selected experiment does not exist."
    experiments = SETUP_FUNCS[args.experiment]()
    with open(args.o, "w", newline="") as f:
        json.dump(list(map(lambda x: x.as_dict(), experiments)), f, indent=4)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(prog="Experiment CLI", description="Provides an interface to setup and run experiments")
    subparsers = parser.add_subparsers(required=True)

    run_parser = subparsers.add_parser("run", help="Runs an experiment. The setup json is expected on stdin")
    run_parser.set_defaults(func=run)
    run_parser.add_argument("-o", default=str(REPO_DIR / f"results_{timestamp}.json"))

    setup_parser = subparsers.add_parser("setup", help="Setup an experiment by filling a json with run descriptions")
    setup_parser.set_defaults(func=setup)
    setup_parser.add_argument("experiment", choices=SETUP_FUNCS.keys())
    setup_parser.add_argument("-o", default=str(REPO_DIR / "experiments.json"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
