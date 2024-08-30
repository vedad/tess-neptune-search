#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1

module purge
module restore

srun python search.py --target-list /home/astro/phsrmj/transit-search/data/sample_files_1000.txt --save /home/astro/phsrmj/transit-search/results/tests/20230808_biweight --period-max 12
