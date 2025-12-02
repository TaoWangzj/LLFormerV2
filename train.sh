#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:./

torchrun --nnodes=1 --nproc_per_node=2 --master_port=4222 basicsr/train.py \
    --option options/train/LLFormerv2+_UHD.yml \
    --launcher pytorch