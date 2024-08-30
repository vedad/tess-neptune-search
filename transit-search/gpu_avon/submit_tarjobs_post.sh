#!/bin/bash

for index in {1..19}; do
    printf -v dir_id "%03d" $index
    sbatch submit_tarjob.sbatch "run_${dir_id}"
done
