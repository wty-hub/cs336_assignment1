#!/bin/bash

# 要搜索的学习率列表
## 请注意这里 learning_rates 是字符串的数组
learning_rates=(1e-1 1e-2 1e-3 1e-4)

echo "Starting Learning Rate Search..."

## 这里 learning_rates[@] 表示learning_rates的所有元素，
## ${learning_rates[@]}表示取所有元素，加双引号防止数组元素内有空格时出现错误
for lr in "${learning_rates[@]}"
do
    echo "=================================================="
    echo "Running training with Learning Rate: $lr"
    echo "=================================================="
    
    # 为每个实验设置唯一的运行名称和输出目录，防止覆盖
    run_name="lr_search_${lr}"
    out_dir="checkpoints/${run_name}"
    
    # 运行训练脚本
    uv run cs336_basics/train/train_lm.py \
        --lr "$lr" \
        --out_dir "$out_dir" \
        --max_steps 5000 \
        --eval_interval 200 \
        --warmup_steps 100 \
        --log_interval 50 \
        
    echo "Finished run for LR: $lr"
    echo ""
done

echo "All experiments completed."

