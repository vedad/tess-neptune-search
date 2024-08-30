#!/usr/bin/env python

from pathlib import Path

p = Path("/home/astro/phsrmj/transit-search/data")
filepaths = sorted(list(p.glob("sample_files_5k_*.txt")))

for i,fi in enumerate(filepaths):
    script_name = f"/home/astro/phsrmj/transit-search/gpu/slurm_scripts/run_{i:03d}.sbatch"
    savedir = f"/home/astro/phsrmj/transit-search/results/injected/run_{i:03d}"
    with open(script_name, "w") as handle:
        print(f"""#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1

module purge
module restore

srun python search.py --target-list {fi.as_posix()} --save {savedir} --period-max 12 --qmin-fac 0.5 --qmax-fac 2 --dlogq 0.05 --period-min 0.5 --oversampling 5 --window-length 0.5 --detrending-method biweight
                """, file=handle)

