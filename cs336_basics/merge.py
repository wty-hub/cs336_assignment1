import heapq
import pickle
from typing import Iterable

from cs336_basics.single_pretokenization import single_pretokenize_iter


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


class TokenNode:
    """单词词元所构成链表的节点"""

    def __init__(self, token: bytes, word_status):
        self.token = token
        self.prev = None
        self.next = None
        # 是否有效（无效就是被合并了）
        self.alive = True
        self.word_status = word_status
        self.left_bigram = None
        self.right_bigram = None

    def link_prev(self, node):
        self.prev = node
        if node is not None:
            node.next = self

    def link_next(self, node):
        self.next = node
        if node is not None:
            node.prev = self

    def get_next(self):
        nxt = self.next
        while nxt is not None and not nxt.alive:
            nxt = nxt.next
        return nxt

    def get_prev(self):
        prev = self.prev
        while prev is not None and not prev.alive:
            prev = prev.prev
        return prev

    def refer(self, bigram, from_left: bool):
        if from_left:
            self.bigrams[0] = bigram
        else:
            self.bigrams[1] = bigram

    def decrease_the_other_bigram(self, left: bool):
        other_bigram = self.left_bigram if left else self.right_bigram
        other_bigram.decrease_freq(self)

    def die(self):
        self.alive = False

    def get_freq(self):
        return self.word_status.freq


class WordStatus:
    """单词的当前状态，包括token和出现频率"""

    def __init__(self, word: bytes):
        self.word = word
        # token 链表的头
        self.token_node_head = None
        tail = None
        for c in word:
            t = bytes([c])
            new_node = TokenNode(t, self)
            if self.token_node_head is None:
                self.token_node_head = new_node
                tail = new_node
            else:
                tail.link_next(new_node)
                tail = new_node
        self.freq = 0

    def merge_nodes(self, l_node: TokenNode, r_node: TokenNode):
        """合并 l_node 和 r_node, 返回新的 node"""
        assert l_node.get_next() is r_node
        if not l_node.alive or not r_node.alive:
            return None
        new_node = TokenNode(l_node.token + r_node.token, self)
        # 如果 l_node 是链表头结点，则需要重新将链表头指向 new_node
        if self.token_node_head is l_node:
            self.token_node_head = new_node
        else:
            prev_node: TokenNode = l_node.get_prev()
            # 这里不会为 None
            prev_node.link_next(new_node)
        next_node = r_node.get_next()
        new_node.link_next(next_node)
        l_node.die()
        r_node.die()
        return new_node

    def count(self):
        self.freq += 1

    def __eq__(self, value):
        return value.word == self.word

    def __hash__(self):
        return hash(self.word)

    def __str__(self):
        return f'[WordStatus "{self.word}"]'

    def __repr__(self):
        return self.__str__()


class BigramMeta:
    """双词组的信息"""

    def __init__(self, l_token: bytes, r_token: bytes):
        self.l_token = l_token
        self.r_token = r_token
        # 同时记录 a 和 b 的对应 TokenNode
        self.occurs: set[tuple[TokenNode, TokenNode]] = set()
        self.version = 0
        self.merged = False
        self.freq = 0

    def occur(self, l_node: TokenNode, r_node: TokenNode):
        assert l_node.token == self.l_token
        assert r_node.token == self.r_token
        assert l_node.get_next() is r_node
        if (l_node, r_node) not in self.occurs:
            self.occurs.add((l_node, r_node))
            self.freq += l_node.get_freq()

    def get_new_token(self):
        """合并时获取需要加入词汇表的新token"""
        return self.l_token + self.r_token

    def decrease_freq(self, token_node: TokenNode):
        self.freq -= token_node.get_freq()
        self.version += 1

    def decrease_neighbor_freq(self):
        for l_node, r_node in self.occurs:
            # 这里的左右是反的
            l_node.decrease_the_other_bigram(self, False)
            r_node.decrease_the_other_bigram(self, True)
        self.merged = True

    def __eq__(self, value):
        return (self.l_token, self.r_token) == (value.a_token, value.b_token)

    def __hash__(self):
        return hash((self.l_token, self.r_token))

    def __lt__(self, value):
        if self.freq != value.freq:
            return -self.freq < -value.freq
        return self.concated_token > value.concated_token

    def __str__(self):
        return f"BigramStatus ({self.l_token}, {self.r_token})"

    def __repr__(self):
        return self.__str__()


class HeapElement:
    """双词组频率堆的元素"""

    def __init__(self, bigram_meta: BigramMeta):
        self.bigram_meta = bigram_meta
        self.freq = bigram_meta.freq
        self.version = bigram_meta.version

    def __lt__(self, value):
        return (self.freq, self.pair) > (value.freq, value.pair)


class FastMerger:
    def __init__(self, pretoken_iter: Iterable[bytes]):
        self.word_status_set: set[WordStatus] = set()
        # token集合，一开始有 0~255 的单个字节
        self.token_set: set[bytes] = set(bytes([c]) for c in range(256))
        # 双词组元数据的集合
        self.bigram_meta_dict = dict[tuple[bytes, bytes], BigramMeta] = dict()
        # 初始化所有单词（预词元）状态
        word_dict = dict()
        for w in pretoken_iter:
            if w not in word_dict:
                word_status = WordStatus(w)
                word_dict[w] = word_status
                self.word_status_set.add(word_status)
            else:
                word_status = word_dict[w]
            word_status.count()

        # 初始化所有 BigramMeta
        for ws in self.word_status_set:
            node = ws.token_node_head
            if node is not None:
                next_node = node.get_next()
                while next_node is not None:
                    self.count_bigram(node, next_node)
                    node = next_node
                    next_node = node.get_next()

        self.bigram_heap: list[HeapElement] = []
        # 初始化双词组频率堆
        for bm in self.bigram_meta_dict.values():
            self.bigram_heap.append(HeapElement(bm))
        heapq.heapify(self.bigram_heap)

    def count_bigram(self, l_node: TokenNode, r_node: TokenNode):
        """合并后出现新的node对，更新其所对应的双词组元数据"""
        new_bigram = (l_node.token, r_node.token)
        if new_bigram in self.bigram_meta_dict:
            bigram_meta = self.bigram_meta_dict[new_bigram]
        else:
            bigram_meta = BigramMeta(l_node, r_node)
            self.bigram_meta_dict[new_bigram] = bigram_meta
        bigram_meta.occur(l_node, r_node)

    def merge_once(self):
        while True:
            if len(self.bigram_heap) < 1:
                raise RuntimeError("bigram heap is empty")
            top_element = heapq.heappop(self.bigram_heap)
            # 如果未被合并且为最新版本，则数据有效
            if (
                not top_element.bigram_meta.merged
                and top_element.version == top_element.bigram_meta.version
            ):
                break

        to_merge = top_element.bigram_meta
        self._do_merge(to_merge)

    def _do_merge(self, bigram_meta: BigramMeta):
        if bigram_meta.merged:
            # 确保不会重复合并
            return
        # 第一步：减少相邻双词组的频率
        bigram_meta.decrease_neighbor_freq()
        # 第二步：合并与该双词组相连的node
        new_nodes: list[TokenNode] = []
        for l_node, r_node in bigram_meta.occurs:
            word_status: WordStatus = l_node.word_status
            new_node = word_status.merge_nodes(l_node, r_node)
            new_nodes.append(new_node)
        self.token_set.add(bigram_meta.get_new_token())
        # 第三步：创建或更新 bigram
        for new_node in new_nodes:
            prev_node = new_node.get_prev()
            if prev_node is not None:
                self.count_bigram(prev_node, new_node)
            next_node = new_node.get_next()
            if next_node is not None:
                self.count_bigram(new_node, next_node)

    def get_vocab(self):
        return dict((i, token) for i, token in enumerate(self.token_set))

    def __len__(self):
        return len(self.token_set)


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    print(f"预分词中")
    pre_iter = single_pretokenize_iter(input_path, special_tokens, b"<|endoftext|>")
    fast_merger = FastMerger(pre_iter)
    print(fast_merger.token_set)
    print(fast_merger.bigram_meta_dict)
    print("HALT")