#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29600}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PORT=$((12000 + $RANDOM % 20000))

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# NCCL_DEBUG=INFO torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
