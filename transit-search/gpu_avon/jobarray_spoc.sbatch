#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --array=0-1

module purge
module restore

source_dir="/home/shared/astro/tess/spoc_ffi"
save_dir="/home/shared/astro/tess/spoc_ffi/results"
inputfiles=()
outputdirs=()
indexfiles=()

for number in {0000..0001}; do
    # Create the input and output filenames
    load_filename="${source_dir}/target_list/tic_list_10k_${number}.txt"
    index_file="${source_dir}/target_list/index_file_10k_${number}.pickle"
    save_dirname="${save_dir}/run_${number}"

    # add to array
    inputfiles+=($load_filename)
    indexfiles+=($index_file)
    outputdirs+=($save_dirname)
done

srun python search.py --target-list ${inputfiles[${SLURM_ARRAY_TASK_ID}]} --index-file ${indexfiles[${SLURM_ARRAY_TASK_ID}]} --save ${outputdirs[${SLURM_ARRAY_TASK_ID}]} --period-max 16 --qmin-fac 0.5 --qmax-fac 2 --dlogq 0.02 --period-min 0.5 --oversampling 5 --window-length 0.5 --detrending-method biweight
