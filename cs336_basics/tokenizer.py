from collections import defaultdict
from dataclasses import dataclass
import heapq
import json
from typing import Iterable, Optional

from ordered_set import OrderedSet
import regex

from tests.common import gpt2_bytes_to_unicode


class PreTokenizer:
    """预分词类，以便于使用"""

    # 经典的 GPT-2 分词正则
    PAT = (
        rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    # 编译以提升复用效率
    CPD_PAT = regex.compile(PAT)

    def __init__(self, text: bytes):
        self.text = text

    def iter(self):
        for s in PreTokenizer.CPD_PAT.finditer(self.text):
            yield s.group(0)

    @classmethod
    def iter_text(cls, text: str):
        for s in cls.CPD_PAT.finditer(text):
            yield s.group(0)


class TokenNode:
    __slots__ = ("token", "prev", "next", "alive")

    def __init__(self, token: bytes):
        self.token = token
        self.prev = None
        self.next = None
        self.alive = True

    def insert_prev(self, node):
        p_prev = self.prev
        if p_prev is not None:
            p_prev.next = node
        node.prev = p_prev
        node.next = self
        self.prev = node

    def insert_next(self, node):
        p_next = self.next
        if p_next is not None:
            p_next.prev = node
        node.prev = self
        node.next = p_next
        self.next = node

    def get_next(self):
        n = self.next
        while n is not None and not n.alive:
            n = n.next
        self.next = n
        return n

    def get_prev(self):
        p = self.prev
        while p is not None and not p.alive:
            p = p.prev
        self.prev = p
        return p

    def die(self):
        self.alive = False

    def get_node_pairs(self):
        """获取当前结点属于的所有结点对"""
        res: list[NodePair] = []
        prev = self.get_prev()
        if prev is not None:
            res.append((prev, self))
        nxt = self.get_next()
        if nxt is not None:
            res.append((self, nxt))
        return res

    @staticmethod
    def build_list(word: bytes):
        """创建链表，每个byte一个结点"""
        head = TokenNode(b"")
        tail = TokenNode(b"")
        head.insert_next(tail)
        for c in word:
            b = bytes([c])
            new_node = TokenNode(b)
            tail.insert_prev(new_node)
        return head, tail


TokenPair = tuple[bytes, bytes]
NodePair = tuple[TokenNode, TokenNode]


@dataclass(order=True)
class HeapElement:
    # 小根堆元素，只记录 token 和优先级 rank
    rank: int
    pair: TokenPair


def node_pair_valid(pair: NodePair):
    return pair[0].alive and pair[1].alive


def merge_node(pair: NodePair):
    assert pair[0].get_next() is pair[1]
    new_node = TokenNode(pair[0].token + pair[1].token)
    pair[0].insert_next(new_node)
    pair[0].die()
    pair[1].die()
    return new_node


def get_token_pair(node_pair: NodePair) -> TokenPair:
    return (node_pair[0].token, node_pair[1].token)


class PretokenSplitter:
    """负责将 pretoken 分割为最优的 token，以便编码"""

    def __init__(self, merges: list[TokenPair]):
        # 没有 special_tokens，因为预分词的时候去掉了
        self.merges = merges
        # 越靠前的 pair 优先级越高
        self.ranks: dict[TokenPair, int] = dict()
        for idx, tple in enumerate(merges):
            self.ranks[tple] = idx
        # 每个双词组对应的 node
        self.pair_to_nodes: defaultdict[TokenPair, set[NodePair]] = defaultdict(set)
        # 我发现很多单词都是重复的，可以使用缓存机制
        self.cache: dict[bytes, list[bytes]] = dict()

    def _pair_can_be_merged(self, pair: TokenPair):
        return pair in self.ranks

    def _bind_node_pair(self, node_pair: NodePair):
        self.pair_to_nodes[get_token_pair(node_pair)].add(node_pair)

    def _get_node_pairs(self, pair: TokenPair):
        return self.pair_to_nodes[pair]

    def _add_to_heap(self, new_pair: TokenPair, heap: list[HeapElement]):
        if self._pair_can_be_merged(new_pair):
            heapq.heappush(heap, HeapElement(self.ranks[new_pair], new_pair))

    def split(self, pretoken: bytes):

        if pretoken in self.cache:
            return self.cache[pretoken]
        # 边界情况：长度小于等于1
        if len(pretoken) <= 1:
            return [pretoken]

        # 初始化
        ## 清理状态
        self.pair_to_nodes.clear()
        ## 建立链表
        head, tail = TokenNode.build_list(pretoken)
        if head.get_next() is tail:
            return []

        ## 初始化每个双词组所对应的链表节点
        node = head.get_next()
        while node is not tail and node.get_next() is not tail:
            pair = (node.token, node.get_next().token)
            self._bind_node_pair((node, node.get_next()))
            node = node.get_next()

        ## 初始化堆
        heap: list[HeapElement] = []
        for pair in self.pair_to_nodes:
            if self._pair_can_be_merged(pair):
                rank = self.ranks[pair]
                elem = HeapElement(rank, pair)
                heap.append(elem)
        heapq.heapify(heap)

        # 合并过程
        while len(heap) > 0:
            top = heapq.heappop(heap)
            self._merge_token_pair(top.pair, heap)

        res: list[bytes] = []
        # head是哨兵头结点
        node = head.get_next()
        while node is not tail:
            res.append(node.token)
            node = node.get_next()

        self.cache[pretoken] = res
        return res

    def _merge_token_pair(self, token_pair: TokenPair, heap: list[HeapElement]):
        # 获取单词链表中所有对应的结点对
        node_pairs = self.pair_to_nodes[token_pair]
        new_token_pairs: set[TokenPair] = set()
        for node_pair in node_pairs:
            # 逐个合并结点对
            if not node_pair_valid(node_pair):
                continue
            new_node = merge_node(node_pair)
            # 查看合并后出现的新结点对
            new_node_pairs = new_node.get_node_pairs()
            for new_node_pair in new_node_pairs:
                token_pair = get_token_pair(new_node_pair)
                new_token_pairs.add(token_pair)
                # 绑定新的token对与node对
                self._bind_node_pair(new_node_pair)
        for new_token_pair in new_token_pairs:
            self._add_to_heap(new_token_pair, heap)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[TokenPair],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.reverse_vocab = dict((value, key) for key, value in self.vocab.items())
        self.merges = merges
        self.special_tokens = special_tokens
        self.remove_regex = None
        if self.special_tokens is not None:
            pattern = "|".join(regex.escape(s) for s in self.special_tokens)
            self.remove_regex = regex.compile(pattern)
        self.splitter = PretokenSplitter(self.merges)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """根据文件创建新的 Tokenizer"""
        # vocab file 是 json 格式
        ## 原文件中 key 和 value 是颠倒的
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_reversed = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes(
                    [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
                )
                for gpt2_vocab_item, gpt2_vocab_index in vocab_reversed.items()
            }
        # merges 格式是每一行两个被merge的token，用空格隔开
        ## GPT-2 使用了特殊字符表示不可见字符，需要转换
        with open(merges_filepath, "r") as merges_file:
            merges: list[TokenPair]
            gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
            merges_raw = [tuple(line.rstrip().split(" ")) for line in merges_file]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in merges_raw
            ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        res: list[int] = []
        if self.remove_regex is not None:
            text = self.remove_regex.sub("", text)
        text: bytes = text.encode()
        pretokenizer = PreTokenizer(text)
        for pretoken in pretokenizer.iter():
            splits = self.splitter.split(pretoken)
            for split in splits:
                token_id = self.reverse_vocab[split]
                res.append(token_id)
        return res

    def encode_iterable(self, iterable: Iterable[str]):
        for s in iterable:
            s = self.remove_regex.sub("", s)
            text = s.encode()
            for pretoken in PreTokenizer.iter_text(text):
                splits = self.splitter.split(pretoken)
                for split in splits:
                    token_id = self.reverse_vocab[split]
                    yield token_id

    def decode(self, ids: list[int]):
        tokens: list[bytes] = []
        for id in ids:
            tokens.append(self.vocab[id])
        return b"".join(tokens).decode()
