# 计算 Transformer LM resource accounting

# GPT-2 XL 的超参数
vocab_size = 50257
# context_length = 1024
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

p_total = (
    2 * vocab_size * d_model + num_layers * (12 * d_model**2 + 2 * d_model) + d_model
)
print(f"first term: {16 * p_total}")

last_factor = 4 * context_length * (num_layers * (16 * d_model + 2 * num_heads * context_length) + d_model + vocab_size)
print(f'last factor: {last_factor}')