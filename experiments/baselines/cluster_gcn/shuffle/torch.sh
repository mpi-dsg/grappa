#!/bin/bash
# Example distributed training script
# Adjust paths and parameters for your environment

TORCHRUN=$(which torchrun || echo "python3 -m torch.distributed.run")
$TORCHRUN \
  --nproc_per_node=1 \
  --nnodes=4 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=9988 \
  distributed_tensor_shuffle.py
