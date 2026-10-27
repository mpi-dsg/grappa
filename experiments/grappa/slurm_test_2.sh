#!/bin/bash
#SBATCH --partition=h100        # Use GPU partition "a100"
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:1            # set 2 GPUs per job
#SBATCH -c 8                   # Number of core
#SBATCH -t 1-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=97GB            # Memory pool for all cores (see also --mem-per-cpu)

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) #=sws-8a100-03
export MASTER_PORT=9901

echo $MASTER_ADDR $MASTER_PORT

srun --output=$LOG_DIR/%x_%j_task%t_%N.out \
     --error=$LOG_DIR/%x_%j_task%t_%N.err \
     hostname

echo '---'
scontrol show hostnames

