import argparse
import json
from pathlib import Path
from src.methods.base import RunParams
from src.plot import PLOT_FUNCS
from src.util import REPO_DIR


def plot(args):
    assert args.plot in PLOT_FUNCS, "The selected plot does not exist."
    all_results = []
    current_params = set()
    for path in args.infiles:
        path = Path(path)
        assert path.exists()
        with open(path, "r") as json_file:
            results = json.load(json_file)
            new_params = map(lambda x: RunParams(**x["p"]), results)
            old_len = len(current_params)
            new_len = len(results)
            current_params.update(new_params)
            combined_len = len(current_params)
            if combined_len != old_len + new_len:
                print("!--------------------------------------------------!")
                print("!WARNING: Merging results with duplicate RunParams.!")
                print("!--------------------------------------------------!")
            all_results.extend(results)
            
    PLOT_FUNCS[args.plot](all_results, Path(args.o))


def main():
    parser = argparse.ArgumentParser(
        prog="Plot experiment results",
        description="Create a plot from an experiment results json(s)",
    )
    parser.add_argument("plot", choices=PLOT_FUNCS.keys())
    parser.add_argument("infiles", nargs="+", help="The input file(s). If multiple, the results are merged.")
    parser.add_argument("-o", help="Plot output directory", default=str(REPO_DIR))

    args = parser.parse_args()
    plot(args)


if __name__ == "__main__":
    main()
