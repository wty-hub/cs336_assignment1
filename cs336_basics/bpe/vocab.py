from collections import defaultdict
import pickle
from typing import Iterable, Iterator


def init_vocab_freq(pretokenizer_iter: Iterator[bytes]) -> dict[tuple[bytes], int]:
    """获取预词元频率"""
    res = {}
    for p in pretokenizer_iter:
        t = tuple(bytes([b]) for b in p)
        res[t] = res.get(t, 0) + 1
    return res

