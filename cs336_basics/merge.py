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

def substitute(origin: tuple[bytes], bigram_tuple: tuple[bytes]):
    """将origin中的所有符合bigram_tuple的子串替换为单个连接起来的token"""
    res = []
    bigram_bytes = b''.join(bigram_tuple)
    i = 0
    while i < len(origin):
        if i < len(origin) - 1 and origin[i] == bigram_tuple[0] and origin[i+1] == bigram_tuple[1]:
            i+=2
            res.append(bigram_bytes)
        else:
            res.append(origin[i])
            i += 1
    return tuple(res)
    

def slow_merge_once(
    bigram_freq: dict[tuple[bytes], int], vocab_freq: dict[tuple[bytes], int]
):
    """对频率最高的双词组进行合并, 返回新的单词频率vocab_freq"""
    highest_bigram: tuple[bytes] = max(bigram_freq, key=bigram_freq.get)
    new_freq = {}
    for word, freq in vocab_freq.items():
        # 遍历原单词频率，重新构建单词频率
        if is_subtuple(highest_bigram, word):
            new_freq[substitute(word, freq)] = freq
        else:
            new_freq[word] = freq
    return new_freq
        