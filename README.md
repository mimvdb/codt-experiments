# CODT experiments
This repository hosts the experiment setup for CODT, including comparisons against other methods.

## Setup
1. Each method is included as a git submodule, to clone all these run `git submodule update --init` after cloning. If you also want to be able to make changes in the submodules, make them track their main branch with `git submodule foreach git checkout main` and `git submodule foreach git pull origin main`
2. Install [`uv`](https://github.com/astral-sh/uv) (faster pip replacement).

## Running experiments
Run `uv run main.py`.