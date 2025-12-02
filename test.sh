#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./

torchrun --nnodes=1 --nproc_per_node=1 --master_port=4320 basicsr/test.py \
    --option options/test/LLFormerv2+.yml \
    --launcher pytorch