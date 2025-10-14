import pickle
from typing import BinaryIO

from cs336_basics.single_pretokenization import pretokenize_iter
from typing import Iterator


def get_prepair_freq(pretokenizer_iter: Iterator[bytes]) -> dict[tuple[bytes], int]:
    """获取预词元频率"""
    res = {}
    for p in pretokenizer_iter:
        t = tuple(bytes(b) for b in p)
        res[t] = res.get(t, 0) + 1
    return res


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: bytes,
):
    file: BinaryIO = open(input_path, mode="rb")

    # 第一步：词汇表初始化
    ## 使用 byte-level BPE tokenizer, 所以一开始的 vocab size 是 256
    cur_vocab_size = 256
    ## 一开始的 vocab 是所有单字节bytes的集合
    cur_vocab = {bytes([b]) for b in range(256)}

    # 第二步：预分词 + 统计字符对频率
    pre_iter = pretokenize_iter(input_path, special_tokens, split_special_token)
    freqs = get_prepair_freq(pre_iter)


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-train.txt"
    load_path = "tmp/TinyStoriesV2-GPT4-train-pretokens.pkl"
    print(f"loading pickle")
    with open(load_path, "rb") as f:
        pretokens: list[bytes] = pickle.load(f)
    print(f"counting frequency")
    freq = get_prepair_freq(iter(pretokens))
    save_path = "tmp/TinyStoriesV2-GPT4-train-freqs.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(freq, f)
    # print(freq)
    # print(f'get {len(freq.keys())} tokens and {len(set(freq.values()))} kind of values')
