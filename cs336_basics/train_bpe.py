import pickle
from typing import BinaryIO

from cs336_basics.bigram import get_bigram_freq
from cs336_basics.merge import get_highest_bigram, slow_merge_once
from cs336_basics.single_pretokenization import single_pretokenize_iter
from typing import Iterator
from tqdm import tqdm

from tests.common import gpt2_bytes_to_unicode


def get_prepair_freq(pretokenizer_iter: Iterator[bytes]) -> dict[tuple[bytes], int]:
    """获取预词元频率"""
    res = {}
    for p in pretokenizer_iter:
        t = tuple(bytes([b]) for b in p)
        res[t] = res.get(t, 0) + 1
    return res


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: bytes = b"<|endoftext|>",
):
    # 第一步：词汇表初始化
    ## 使用 byte-level BPE tokenizer, 所以一开始的 vocab size 是 256
    ## 一开始的 vocab 是所有单字节bytes的集合加上所有特殊token
    cur_vocab_set = set()
    cur_vocab_set.update(t.encode() for t in special_tokens)
    cur_vocab_set.update(bytes([b]) for b in range(256))

    # 第二步：预分词, 统计单词频率和双词组频率
    print(f"预分词中")
    pre_iter = single_pretokenize_iter(input_path, special_tokens, split_special_token)
    vocab_freq = get_prepair_freq(pre_iter)
    bigram_freq = get_bigram_freq(vocab_freq)

    # 第三步：训练，不断增大当前词汇量
    num_merges = vocab_size - len(cur_vocab_set)
    pbar = tqdm(total=num_merges, desc="BPE 训练中")
    merges = []
    while len(cur_vocab_set) < vocab_size:
        ## 获取当前频率最高的双词组
        highest_bigram = get_highest_bigram(bigram_freq)
        ## 更新词汇表
        new_token = b"".join(highest_bigram)
        cur_vocab_set.add(new_token)
        merges.append(highest_bigram)
        ## 更新单词频率
        vocab_freq = slow_merge_once(vocab_freq, highest_bigram)
        ## 更新双词组频率
        bigram_freq = get_bigram_freq(vocab_freq)
        bigram_count = len(bigram_freq.keys())
        pbar.update(1)
        if bigram_count <= 1:
            ## 边界情况：没有双词组可供合并
            print("无法继续合并，退出")
    pbar.close()

    # 第四步：根据要求，返回词汇表和合并记录
    vocab = {index: value for index, value in enumerate(cur_vocab_set)}
    return vocab, merges


if __name__ == "__main__":
    train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
