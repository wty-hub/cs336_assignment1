#!/bin/bash

batch_sizes=(1 16 64 128)

echo "Starting Learning Rate Search..."

## 这里 batch_sizes[@] 表示learning_rates的所有元素，
## ${batch_sizes[@]}表示取所有元素，加双引号防止数组元素内有空格时出现错误
for batch_size in "${batch_sizes[@]}"
do
    echo "=================================================="
    echo "Running training with batch size: $batch_size"
    echo "=================================================="
    
    # 为每个实验设置唯一的运行名称和输出目录，防止覆盖
    run_name="${batch_size}"
    out_dir="checkpoints/${run_name}"
    
    # 运行训练脚本
    uv run cs336_basics/train/train_lm.py \
        --batch_size "$batch_size" \
        --lr 1e-3 \
        --out_dir "$out_dir" \
        --max_steps 5000 \
        --eval_interval 200 \
        --warmup_steps 100 \
        --log_interval 50 \
        
    echo "Finished run for batch_size: $batch_size"
    echo ""
done

echo "All experiments completed."

