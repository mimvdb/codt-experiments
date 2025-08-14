# CODTree experiments
This repository hosts the experiment setup for CODTree, including comparisons against other methods.

## Setup
1. Each method is included as a git submodule, to clone all these run `git submodule update --init` after cloning. If you also want to be able to make changes in the submodules, make them track their main branch with `git submodule foreach git checkout main` and `git submodule foreach git pull origin main`
2. Install [`uv`](https://github.com/astral-sh/uv) (faster pip replacement).

## Running experiments
Start with running `uv run util.py quant_to_csv` to generate the required datasets, then `uv run setup.py` to setup an experiment, then `uv run run.py < experiments.json` to run it.

To combine multiple runs of experiments, `jq -s 'flatten(1)' results_*.json > results_aggregated.json` may be useful.