

import torch
from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.util.generate import generate


def generate_text(tokenizer: Tokenizer,
                  lm: TransformerLM,
                  origin_text: str,
                  max_new_tokens: int,
                  temperature: float,
                  eos_special_token: str):
    """
    使用模型生成文本，输入初始的文本，返回初始的 + 生成的文本

    Args:
        tokenizer (Tokenizer): 分词器
        lm (TransformerLM): 训练过的模型
        origin_text (str): 初始的文本（即prompt）
        max_new_tokens (int): 最多生成的token数
        temperature (float): 采样温度，越高越随机，0.0表示贪心输出
        eos_special_token (str): EOS token的字符串表示

    """
    input_ids = torch.tensor(tokenizer.encode(
        origin_text), dtype=torch.long).unsqueeze(0)
    assert eos_special_token in tokenizer.special_tokens
    eos_token_id = tokenizer.encode(eos_special_token)[0]
    output_ids = generate(lm, input_ids, max_new_tokens,
                          temperature, eos_token_id)
    return tokenizer.decode(output_ids[0].tolist())
