#!/bin/bash

cd /lustre/fswork/projects/rech/gax/ums98bp/descriptor_analysis/slurm
WORK_DIR="../mace-0b3"

# Submit first job for mptrj
job1=$(sbatch --export=WORK_DIR=$WORK_DIR,DATASET="mptrj" calculate_descriptors.slurm)
job1_id=$(echo $job1 | awk '{print $4}')  # Extract job ID

# Submit second job for salex
job2=$(sbatch --export=WORK_DIR=$WORK_DIR,DATASET="salex" calculate_descriptors.slurm)
job2_id=$(echo $job2 | awk '{print $4}')  # Extract job ID

# Submit pca.slurm job, dependent on the successful completion of job1 and job2
sbatch --dependency=afterok:$job1_id:$job2_id --export=WORK_DIR=$WORK_DIR pca.slurm