#!/bin/bash

job=$1
for index in {0..19}; do
#for index in {0..9}; do
    job_id="${job}_${index}"
    printf -v dir_id "%03d" $index
    sbatch -d afterok:${job_id} submit_tarjob.sbatch "run_${dir_id}"
done
