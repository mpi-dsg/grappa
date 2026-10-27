#!/bin/bash
#SBATCH --partition=a100        # Use GPU partition "a100"
#SBATCH --gres=gpu:8            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 0-01:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=1600GB            # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclusive            # Request exclusive access to the node
#SBATCH -o $WORK_DIR/exp/cluster_gcn/ckpt/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e $WORK_DIR/exp/cluster_gcn/ckpt/%x_%j.err      # File to which STDERR will be written

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) #=sws-8a100-03
export MASTER_PORT=9901
export GPUS_PER_NODE=4

export HF_HUB_TOKEN=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

srun --jobid $SLURM_JOBID bash -c 'echo $MASTER_ADDR; \
python train.py --dataset ogbpr \
--data_prefix ./data/ \
--nomultilabel \
--num_layers 2  --hidden1 128 --learning_rate 0.003 \
--num_clusters 1500 --bsize 20 \
--dropout 0.5 --weight_decay 0.0001  --early_stopping 1000 \
--num_clusters_val 20 --num_clusters_test 1 \
--epochs 500 \
--save_name ./ckpt/ogbprmodel \
--diag_lambda 0.0001 --novalidation'
