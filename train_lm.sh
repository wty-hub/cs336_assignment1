#!/bin/bash

# 设置学习率
lr=1e-3

run_name='1e-3_test'

# 为每个实验设置唯一的运行名称和输出目录，防止覆盖
run_name="lr_search_${lr}"
out_dir="checkpoints/${run_name}"

# 运行训练脚本
uv run cs336_basics/train/train_lm.py \
    --lr "$lr" \
    --out_dir "$out_dir" \
    --max_steps 80000 \
    --eval_interval 200 \
    --warmup_steps 1000 \
    --log_interval 100 \
    
echo "Finished run for LR: $lr"
