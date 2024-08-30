#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3700

scenario="PLA"
source_dir=$1
data_dir="/home/shared/astro/tess/injected/results/Uniform/$scenario"
run="20240715"

tar -cf "${data_dir}/tarfiles/${run}/${source_dir}.tar" -C $data_dir $source_dir && cd "${data_dir}/${source_dir}" && rm *.fits
