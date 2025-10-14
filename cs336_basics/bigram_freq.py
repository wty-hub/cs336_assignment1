import pickle


def get_bigram_freq(token_freq: dict[tuple[bytes], int]):
    """根据词汇频率，获取双词组的频率"""
    res = {}
    for tokens, freq in token_freq.items():
        for a, b in zip(tokens, tokens[1:]):
            res[(a, b)] = res.get((a, b), 0) + freq
    return res


if __name__ == "__main__":
    load_path = "tmp/TinyStoriesV2-GPT4-train-freqs.pkl"
    with open(load_path, "rb") as f:
        token_freq = pickle.load(f)
    save_path = "tmp/TinyStoriesV2-GPT4-train-origin_bigramfreq.pkl"
    bigram_freq = get_bigram_freq(token_freq)
    with open(save_path, "wb") as f:
        pickle.dump(bigram_freq, f)
    print(f"save origin bigram freq to {save_path}")
