#!/bin/bash

# 设置学习率
lr=1e-3

run_name='test_generate'

out_dir="checkpoints/${run_name}"

# 运行训练脚本
uv run cs336_basics/train/train_lm.py \
    --lr "$lr" \
    --out_dir "$out_dir" \
    --max_steps 160000 \
    --eval_interval 1000 \
    --warmup_steps 2000 \
    --log_interval 200 \
    --batch_size 16 \
    --save_steps 10000
    
echo "Finished run"
