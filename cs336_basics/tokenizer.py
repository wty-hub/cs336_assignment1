from collections import defaultdict
from dataclasses import dataclass
import heapq
import json
from typing import Optional

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

    def merge_with_next(self):
        self.die()
        nxt = self.get_next()
        assert nxt is not None
        nxt.die()
        new_node = TokenNode(self.token + nxt.token)
        self.insert_next(new_node)
        return new_node

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


class HeapElement:
    # 小根堆元素
    def __init__(self, rank):
        self.rank = rank
        self.left_nodes: list[TokenNode] = []
        self.right_nodes: list[TokenNode] = []

    def occur(self, left_node: TokenNode, right_node: TokenNode):
        self.left_nodes.append(left_node)
        self.right_nodes.append(right_node)

    def __lt__(self, value):
        return self.rank < value.rank


class PretokenSplitter:
    """负责将 pretoken 分割为最优的 token，以便编码"""

    def __init__(self, merges: list[tuple[bytes, bytes]]):
        # 没有 special_tokens，因为预分词的时候去掉了
        self.merges = merges
        # 越靠前的 pair 优先级越高
        self.ranks: dict[tuple[bytes, bytes], int] = dict()
        for idx, tple in enumerate(merges):
            self.ranks[tple] = idx
        # 我发现很多单词都是重复的，可以使用缓存机制
        self.cache: dict[bytes, list[bytes]] = dict()

    def heappop_once(self, heap: list[HeapElement]):
        new_node_pairs: set[tuple[TokenNode, TokenNode]] = set()
        top = heapq.heappop(heap)
        for l_node, r_node in zip(top.left_nodes, top.right_nodes):
            if not (l_node.alive and r_node.alive):
                continue
            new_node = l_node.merge_with_next(r_node)
            if new_node.get_prev() is not None:
                new_node_pairs.add((new_node.get_prev(), new_node))
            if new_node.get_next() is not None:
                new_node_pairs.add((new_node, new_node.get_next()))
        self.deal_new_node_pairs(new_node_pairs, heap)

    def deal_new_node_pairs(
        self, new_node_pairs: set[tuple[TokenNode, TokenNode]], heap: list[HeapElement]
    ):
        new_token_pairs: defaultdict[
            tuple[bytes, bytes], list[tuple[TokenNode, TokenNode]]
        ] = defaultdict(list)
        for l_node, r_node in new_node_pairs:
            new_token_pair = (l_node.token, r_node.token)
            if new_token_pair in self.ranks:
                new_token_pairs[new_token_pair].append((l_node, r_node))
        for token_pair, relative_nodes in new_token_pairs.item():
            pass

    def split(self, pretoken: bytes):
        if pretoken in self.cache:
            return self.cache[pretoken]
        # 边界情况：长度小于等于1
        if len(pretoken) <= 1:
            return [pretoken]

        # 初始化
        pair_dict: defaultdict[
            tuple[bytes, bytes], list[tuple[TokenNode, TokenNode]]
        ] = defaultdict(list)
        head, tail = TokenNode.build_list(pretoken)
        node = head
        while node is not None and node.get_next() is not None:
            pair = (node.token, node.get_next().token)
            pair_dict[pair].append(node, node.get_next())
        heap: list[HeapElement] = []
        for pair, occurs in pair_dict.items():
            rank = self.ranks[pair]
            elem = HeapElement(rank)
            for l_node, r_node in occurs:
                elem.occur(l_node, r_node)
            heap.append(elem)
        heapq.heapify(heap)

        while len(heap) > 0:
            self.heappop_once(heap)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.remove_regex = None
        if self.special_tokens is not None:
            pattern = "|".join(regex.escape(s) for s in self.special_tokens)
            self.remove_regex = regex.compile(pattern)

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
            merges: list[tuple[bytes, bytes]]
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
        if self.remove_regex is not None:
            text = self.remove_regex.sub("", text)
        text: bytes = text.encode()
        pretokenizer = PreTokenizer(text)
