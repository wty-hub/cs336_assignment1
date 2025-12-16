import math
import torch
from cs336_basics.transformer.softmax import softmax
from cs336_basics.transformer.transformer_lm import TransformerLM


def decoding(
    lm: TransformerLM,
    idx: torch.Tensor,
    eos_id: int,
    maximum_tokens: int = 0,
    temperature: float = 0.7,
    top_p: int = 0,
):
    """
    产生新的token

    :param lm: 模型
    :param idx: 输入
    :param eos_id: end_of_text token 的 index
    :param maximum_tokens: 最多输出的 token 数量，0为无限
    :param temperature: 温度参数，0为完全确定的输出
    :param top_p: top-p 采样参数，0为不截断
    """

    cnt = 0
    while maximum_tokens == 0 or cnt < maximum_tokens:
        cnt += 1
        # 获取 logits 输出
        logits = lm(idx)  # (..., seq_length, vocab_size)
        # 取最后一个位置的 logits
        logits = logits[..., -1, :]  # (..., vocab_size)
        # 施加温度参数
        ## 如果温度为0, 直接输出 logit 最高的 token
        if math.isclose(temperature, 0.0):
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # (..., 1)
        else:
            logits = logits / temperature
            # 对最后一维做 softmax
            if top_p > 0:
                # 获取最大的 k 个值
                top_p_v, _ = torch.topk(logits, top_p)
                # top_p 的最后一个值，即第 p 个
                last = top_p_v[:, -1].unsqueeze(-1)
                # 将小于第 p 个的值改为 -inf，这样其 softmax 输出为0
                logits = torch.where(
                    logits < last,
                    torch.tensor(-float("inf"), device=logits.device),
                    logits,
                )

            probs = softmax(logits, -1)
            # 采样
            next_idx = torch.multinomial(probs, 1)  # (1,)
        idx = torch.cat([idx, next_idx], dim=-1)
        if torch.all(next_idx == eos_id):
            break
        
    return idx
