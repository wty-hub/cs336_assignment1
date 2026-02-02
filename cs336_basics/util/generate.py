import torch
from cs336_basics.transformer.transformer_lm import TransformerLM


def generate(lm: TransformerLM, input_ids, max_new_tokens, temperature, eos_token_id):
    for _ in range(max_new_tokens):
        # 截取超过context length的序列
        idx_cond = (
            input_ids
            if input_ids.size(1) <= lm.config["context_length"]
            else input_ids[:, -lm.config["context_length"]:]
        )
        # 获取最后一个token的logits
        logits = lm(idx_cond)
        logits = logits[:, -1, :]

        if abs(temperature) < 1e5:  # 排除小值，防止数值不稳定
            # 如果 temperature 为 0，那么就是贪心搜索
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            # 使用 softmax 将 logits 转化为概率
            probs = torch.softmax(logits, dim=-1)
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
        # 附加与原始输入之后
        input_ids = torch.cat((input_ids, idx_next), dim=1)
        # 如果得到 eos 则提前退出
        if eos_token_id is not None:
            if (idx_next == eos_token_id).all():
                break

    return input_ids
