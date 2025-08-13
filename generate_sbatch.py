import argparse
import datetime
import json
from pathlib import Path
import numpy as np
import sys
from src.util import REPO_DIR


def run(args):
    experiments = json.loads(Path(args.i).read_bytes())
    max_mem_mb = int(max([e["memory_limit"] for e in experiments]) / 1024 / 1024) + 100 # 100 MB buffer.
    max_timeout = int(max([e["timeout"] for e in experiments]))


    c = args.chunk_size
    seconds_per_chunk = max_timeout * args.chunk_size + 60
    hhmmss = f"{seconds_per_chunk // 3600:02}:{(seconds_per_chunk % 3600) // 60:02}:{seconds_per_chunk % 60:02}"
    total_tasks = int(np.ceil(len(experiments) / c))
    tpj = args.tasks_per_job
    total_jobs = int(np.ceil(total_tasks / tpj))
    if total_jobs == 1:
        tpj = total_tasks
    cpus_per_task = int(np.ceil(max_mem_mb / 4000))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if total_jobs > 100:
        print("WARNING: jobs is greater than 100 and might not be schedulable. Use more tasks per jobs or higher chunk size.", file=sys.stderr)
    if args.tasks_per_job * cpus_per_task > 48:
        print("WARNING: tasks_per_job * cpus_per_task is greater than 48 and might not be schedulable.", file=sys.stderr)

    output_path = Path(args.o)
    output_path.mkdir(parents=True, exist_ok=True)

    array_id = "${SLURM_ARRAY_TASK_ID}"
    srun_lines = []
    for i in range(tpj):
        output = str(output_path / f"results_{timestamp}_{array_id}_{i}.json")
        srun_lines.append(f"srun -c{cpus_per_task} -N1 -n1 --exact uv run run.py -o {output} --chunk-size {c} --chunk-offset $((({array_id} - 1) * {tpj} + {i})) < {args.i} &")
    newline = "\n" # f-string cannot contain backslash

    script = f"""#!/bin/bash
#SBATCH --job-name="CODTree"
#SBATCH --partition=compute-p2
#SBATCH --time={hhmmss} # HH:MM:SS
#SBATCH --ntasks={tpj} # compute has 48 cores, compute-p2 has 64 cores.
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={max_mem_mb//cpus_per_task}M
#SBATCH --account=education-eemcs-msc-cese
#SBATCH --array=1-{total_jobs} # Submit {total_jobs} jobs with ID 1,2,...,{total_jobs}. Education share has max 100 jobs queued
set -x

# Submit this script with `sbatch <script_name>`

# Make sure to not have any (python) modules loaded, so that uv uses its own python download.
# Otherwise the symlink from the venv is broken for different nodes.
# The login node cannot create the environment due to mismatch of python version on compute node, and 
# the compute nodes cannot download packages.
{newline.join(srun_lines)}
wait
"""

    output_path = Path(args.o) / f"sbatch_{timestamp}.sh"
    output_path.write_text(script)
    output_path.chmod(0o755)  # Make the script executable


def main():
    parser = argparse.ArgumentParser(
        prog="SBatch generator",
        description="Generate the sbatch script for running an experiment on SLURM",
    )
    parser.add_argument("-i", default=str(REPO_DIR / "experiments.json"))
    parser.add_argument("-o", default=str(REPO_DIR))
    parser.add_argument('--chunk-size', type=int, default=10, help="The number of runs to batch together in one script invocation")
    parser.add_argument('--tasks-per-job', type=int, default=16, help="The number of tasks to use in one sbatch job")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
