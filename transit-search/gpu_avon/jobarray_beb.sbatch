#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --array=0-19

module purge
module restore

scenario="BEB"
source_dir="/home/shared/astro/tess/injected"
save_dir="/home/shared/astro/tess/injected/results/$scenario"
inputfiles=()
outputdirs=()

# Loop through numbers from 000 to 019
for number in {000..019}; do
    # Create the input and output filenames
    load_filename="${source_dir}/target_list/$scenario/sample_files_5k_${number}.txt"
    save_dirname="${save_dir}/run_${number}"

    # add to array
    inputfiles+=($load_filename)
    outputdirs+=($save_dirname)
done

srun python search.py --target-list ${inputfiles[${SLURM_ARRAY_TASK_ID}]} --save ${outputdirs[${SLURM_ARRAY_TASK_ID}]} --period-max 16 --qmin-fac 0.5 --qmax-fac 2 --dlogq 0.02 --period-min 0.5 --oversampling 5 --window-length 0.5 --detrending-method biweight
