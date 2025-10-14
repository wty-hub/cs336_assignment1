import pickle
from typing import Iterator


def init_vocab_freq(pretokenizer_iter: Iterator[bytes]) -> dict[tuple[bytes], int]:
    """获取预词元频率"""
    res = {}
    for p in pretokenizer_iter:
        t = tuple(bytes([b]) for b in p)
        res[t] = res.get(t, 0) + 1
    return res


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-train.txt"
    load_path = "tmp/TinyStoriesV2-GPT4-train-pretokens.pkl"
    print(f"loading pickle")
    with open(load_path, "rb") as f:
        pretokens: list[bytes] = pickle.load(f)
    print(f"counting frequency")
    freq = init_vocab_freq(iter(pretokens))
    save_path = "tmp/TinyStoriesV2-GPT4-train-freqs.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(freq, f)
