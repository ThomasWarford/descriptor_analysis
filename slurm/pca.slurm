#!/bin/bash
#SBATCH --job-name=pca-faiss
#SBATCH --account gax@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6s
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

echo "Starting job array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"


# set up python env
module purge
module load arch/h100
source ~/.bashrc
mamba activate faiss-torch
export PYTHONPATH="/lustre/fswork/projects/rech/gax/ums98bp/maces/main"
echo "PYTHONPATH=${PYTHONPATH}"



python ../pca.py $WORK_DIR
