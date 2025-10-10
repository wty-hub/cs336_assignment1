from typing import BinaryIO


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    file: BinaryIO = open(input_path, mode="rb")
    
    # 第一步：词汇表初始化
    ## byte-level BPE tokenizer, 所以一开始的 vocab size 是 256
    cur_vocab_size = 256
    ## 一开始的 vocab 是所有单字节bytes的集合
    cur_vocab = {bytes([b]) for b in range(256)}

    # 第二步：