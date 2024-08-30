#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1

#module purge
#module restore

# Define the source directory where your files are located.
source_dir="/home/astro/phsrmj/transit-search/data"
save_dir="/home/astro/phsrmj/transit-search/results/injected"

# Loop through numbers from 0000 to 0309
for number in {000..019}; do
    # Create the output archive filename
    file_name="${source_dir}/sample_files_all_5k_${number}.txt"

    srun python search.py --target-list "$file_name" --save "$save_dir" --period-max 12 --period-min 0.5 --qmin-fac 0.5 --qmax-fac 2 --dlogq 0.05 --oversampling 5 --window-length 0.5 --detrending-method biweight

done
