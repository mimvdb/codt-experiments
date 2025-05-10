import argparse
from src.data import generate_index_sets, quant_to_csv
from src.julia_compile import create_julia_sysimage

UTIL_FUNCS = {"quant_to_csv": quant_to_csv, "generate_index_sets": generate_index_sets, "create_julia_sysimage": create_julia_sysimage}


def util(args):
    assert args.util_func in UTIL_FUNCS, "The selected function does not exist."
    UTIL_FUNCS[args.util_func]()


def main():
    parser = argparse.ArgumentParser(
        prog="Tool to run utility functions", description="Run a utility function"
    )
    parser.add_argument("util_func", choices=UTIL_FUNCS.keys())

    args = parser.parse_args()
    util(args)


if __name__ == "__main__":
    main()
