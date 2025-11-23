context_length = 1024
vocab_size = 50257

def calc(d_model, layers):
    linear_projection_multi_times = 3 * context_length * d_model**2
    print(f"线性投影乘法操作数：{linear_projection_multi_times:.2e}")
    logis_multi_times = context_length**2 * d_model
    print(f"logis计算乘法操作数：{logis_multi_times:.2e}")
    attention_weight_multi_times = context_length**2 * d_model
    print(f"注意力权重计算乘法次数：{attention_weight_multi_times:.2e}")
    MHSA_multiply_times = (
        linear_projection_multi_times + logis_multi_times + attention_weight_multi_times
    )
    print(f"MHSA 乘法操作数：{MHSA_multiply_times:.2e}")

    FFW_multiply_times = 16 * context_length * d_model ** 2
    print(f"FFW 乘法次数{FFW_multiply_times:.2e}")
    Transfomer_block_multiply_times = MHSA_multiply_times + FFW_multiply_times
    print(f"Transformer block 乘法操作数：{Transfomer_block_multiply_times:.2e}")
    Transfomer_total_multiply_times = Transfomer_block_multiply_times * layers
    print(f"Transformer 总乘法操作数：{Transfomer_total_multiply_times:.2e}")

    ln_output_multiply_time = context_length * d_model * vocab_size
    print(f"LN 输出乘法操作数：{ln_output_multiply_time:.2e}")

    total_multiply_time = Transfomer_total_multiply_times + ln_output_multiply_time
    print(f"整体乘法操作数：{total_multiply_time:.2e}")

print('small:')
calc(768, 12)
print('medium:')
calc(1024, 24)
print('large:')
calc(1280, 36)
