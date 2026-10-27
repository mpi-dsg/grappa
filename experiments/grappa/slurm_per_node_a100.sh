#!/bin/bash
#SBATCH --partition=a100        # Use GPU partition "a100"
#SBATCH --nodes 4
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:4            # set 2 GPUs per job
#SBATCH -c 16                   # Number of core
#SBATCH -t 1-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=976GB            # Memory pool for all cores (see also --mem-per-cpu)

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) #=sws-8a100-03
export MASTER_PORT=9901

echo $MASTER_ADDR $MASTER_PORT
echo '-----------------------'

export WHICH_PYTHON=$PYTHON
# slurm 4 gpu per node
export PROC_PER_NODE=4
export N_NODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
#MASTER_ADDR=
#MASTER_PORT=

export DATA_PATH=$DATA_PATH/t10n/tmp/
export GRAPH_NAME=igb260m
export PART_METHOD=random
export NUM_PART=16
export MODEL=gcn
export N_LAYER=2
export FAN_OUT=25,10
export N_EPOCH=51
export BS=1000
export EVAL_BS=200000
export GPU_PER_NODE=4
export N_EVAL=50
export TAG_ID=slurm_per_node_a100


srun --output=$LOG_DIR/%x_%j_task%t_%N.out \
     --error=$LOG_DIR/%x_%j_task%t_%N.err \
$WHICH_PYTHON -m torch.distributed.run \
--nproc_per_node $PROC_PER_NODE --nnodes=$N_NODES --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
train/t10n_train.py \
--data_path $DATA_PATH    \
--graph_name $GRAPH_NAME   \
--part_method $PART_METHOD    \
--n_part $NUM_PART \
--model $MODEL    \
--num_layers $N_LAYER  \
--fan_out $FAN_OUT  \
--num_epochs $N_EPOCH  \
--batch_size $BS \
--batch_size_eval $EVAL_BS \
--stop_at_border \
--disable_backup_server \
--num_gpus $GPU_PER_NODE \
--backend nccl \
--log_every 100 \
--eval_every $N_EVAL \
--socket_ifname bond0 \
--close_dd \
--tag_id $TAG_ID 2>&1 | tee $WORK_ROOT/exp/t10n/log/${TAG_ID}_${NODE_RANK}.log

