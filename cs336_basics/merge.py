import pickle


def is_subtuple(sub: tuple, full: tuple):
    """sub是否是full的子串"""
    n, m = len(sub), len(full)
    assert n != 0
    if n > m:
        return False
    for i in range(m - n + 1):
        if full[i : i + n] == sub:
            return True
    return False


def substitute(origin: tuple[bytes], bigram_tuple: tuple[bytes], bigram_token: bytes):
    """将origin中的所有符合bigram_tuple的子串替换为单个连接起来的token"""
    res = []
    i = 0
    while i < len(origin):
        if (
            i < len(origin) - 1
            and origin[i] == bigram_tuple[0]
            and origin[i + 1] == bigram_tuple[1]
        ):
            i += 2
            res.append(bigram_token)
        else:
            res.append(origin[i])
            i += 1
    return tuple(res)


def get_highest_bigram(bigram_freq: dict[tuple[bytes], int]):
    """获取频率最高的双词组"""
    highest_bigram: tuple[bytes] = max(bigram_freq, key=lambda k: (bigram_freq[k], k))
    return highest_bigram


def slow_merge_once(
    vocab_freq: dict[tuple[bytes], int],
    highest_bigram: tuple[bytes],
):
    """对频率最高的双词组进行合并, 返回新的单词频率 vocab_freq"""
    new_bigram_token = b"".join(highest_bigram)
    new_vocab_freq = {}
    for word, freq in vocab_freq.items():
        # 遍历原单词频率，重新构建单词频率
        if is_subtuple(highest_bigram, word):
            new_vocab_freq[substitute(word, highest_bigram, new_bigram_token)] = freq
        else:
            new_vocab_freq[word] = freq
    return new_vocab_freq


if __name__ == "__main__":
    with open("tmp/TinyStoriesV2-GPT4-train-freqs.pkl", "rb") as f:
        vocab_freq = pickle.load(f)
    with open("tmp/TinyStoriesV2-GPT4-train-origin_bigramfreq.pkl", "rb") as f:
        bigram_freq = pickle.load(f)
