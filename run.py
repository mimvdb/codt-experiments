import argparse
import datetime
import json
import numpy as np
import sys
from src.methods import Run, RunParams, get_method
from src.util import REPO_DIR


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
        method = get_method(params.method, params.task)
        if args.v > 0:
            print("Starting run: ", params)
        output = method.run(params)
        runs.append(Run(params, output).as_dict())
    with open(args.o, "w", newline="") as f:
        json.dump(runs, f, indent=4, default=encode_numpy)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        prog="Experiment runner",
        description="Runs an experiment. The setup json is expected on stdin",
    )
    parser.add_argument("-o", default=str(REPO_DIR / f"results_{timestamp}.json"))
    parser.add_argument('-v', action='count', default=0)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
