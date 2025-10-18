#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A practical Byte-Level BPE trainer core loop:
- mmap zero-copy reading
- regex pretokenizer over bytes
- parallel initial pair counting
- max-heap for top pair
- local (surgical) updates after each merge

Author: you
"""

from __future__ import annotations
import os
import mmap
import heapq
import itertools
from dataclasses import dataclass
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Tuple, List, Dict

import regex  # pip install regex

# ---------------------------
# 预分词：byte-level（GPT-2 风格可替换）
# ---------------------------
# 注意：下面 PAT 是简化示例。按需替换为你的 cs336/GPT2 预分词规则。
PAT = regex.compile(
    rb" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    flags=regex.VERSION1,
)

def pretokenize_bytes(chunk: bytes) -> Iterable[bytes]:
    # 逐 token 产出 bytes 片段
    for m in PAT.finditer(chunk):
        yield m.group(0)

# ---------------------------
# 词元与词表示
# ---------------------------

# 初始词表：0..255 为单字节；后续 merge 产生新 id
BASE_VOCAB_SIZE = 256

@dataclass(frozen=True)
class Pair:
    a: int
    b: int

# 一个词（word）表示为 int id 的列表（初始：字节 → 相同 id）
IntSeq = List[int]

# ---------------------------
# 并行：初次 pair 统计（不包含合并）
# ---------------------------

def _count_pairs_in_tokens(tokens: List[IntSeq]) -> Counter:
    c = Counter()
    for ids in tokens:
        if not ids or len(ids) == 1:
            continue
        # 统计相邻 pair
        prev = ids[0]
        for cur in ids[1:]:
            c[Pair(prev, cur)] += 1
            prev = cur
    return c

def _read_and_tokenize_chunk(mm: memoryview, start: int, end: int) -> List[IntSeq]:
    # 基于 regex 的 bytes 预分词，然后把每个 token 的每个字节作为一个 id（byte-level）
    # 也可替换为“按词”或“按行”的策略
    tokens: List[IntSeq] = []
    view = bytes(mm[start:end])  # 注意：这里复制了分块，可换更细的策略以避免复制
    for tok in pretokenize_bytes(view):
        # byte-level：直接取每个字节的 int 值
        tokens.append(list(tok))
    return tokens

def initial_parallel_tokenize_and_count(path: str, num_workers: int = max(os.cpu_count() or 4, 4)):
    file_size = os.path.getsize(path)
    # 分块：尽量在换行边界处切（简单策略：均分后向后扫到第一个 \n）
    splits = []
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        mv = memoryview(mm)
        chunks = []
        step = max(8 << 20, file_size // num_workers)  # ~8MB 起
        start = 0
        while start < file_size:
            end = min(file_size, start + step)
            # 向后寻找换行，最多前探 64KB
            probe_end = min(file_size, end + 64 * 1024)
            while end < probe_end and end < file_size and mm[end:end+1] != b"\n":
                end += 1
            chunks.append((start, end))
            start = end

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            # 1) 预分词 + byte->int ids
            tokenized_parts = list(ex.map(
                lambda se: _read_and_tokenize_chunk(mv, se[0], se[1]),
                chunks
            ))
            # 2) 初次 pair 统计
            counters = list(ex.map(_count_pairs_in_tokens, tokenized_parts))

    # 汇总
    tokens: List[IntSeq] = list(itertools.chain.from_iterable(tokenized_parts))
    pair_counter = Counter()
    for c in counters:
        pair_counter.update(c)
    return tokens, pair_counter

# ---------------------------
# 维护 pair 出现位置（用于局部更新）
# ---------------------------

def build_pair_positions(tokens: List[IntSeq]) -> Dict[Pair, List[Tuple[int, int]]]:
    """
    返回 pair -> [(word_idx, pos), ...]
    pos 表示 pair 的左元素在该词中的下标。
    """
    pos = defaultdict(list)
    for wi, ids in enumerate(tokens):
        for i in range(len(ids) - 1):
            p = Pair(ids[i], ids[i+1])
            pos[p].append((wi, i))
    return pos

# ---------------------------
# 合并：把 (A,B) 合成新 id，局部更新 tokens / pair_counter / positions
# ---------------------------

def apply_merge(
    tokens: List[IntSeq],
    pair: Pair,
    new_id: int,
    pair_counter: Counter,
    positions: Dict[Pair, List[Tuple[int, int]]],
):
    """
    合并 pair=(a,b) 为新 token new_id。
    仅访问受影响的词及其邻域，局部更新统计。
    """
    a, b = pair.a, pair.b
    locs = positions.get(pair)
    if not locs:
        return

    # 记录本轮被修改的词索引，避免重复处理
    touched_words = set(wi for wi, _ in locs)

    # 为了避免位置在合并中失效，按每个词自左向右重写并重建邻域 pair
    # 同时更新 pair_counter 和 positions：先撤销旧 pair，再加入新 pair
    # 简化：对受影响词做“局部重建”，不去全局扫
    # （进一步优化：可做“差分”更新，这里保持清晰）
    for wi in touched_words:
        ids = tokens[wi]
        if len(ids) < 2:
            continue

        # 一次线性扫描，把 ... a b ... 合并为 ... new_id ...
        out: List[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i+1] == b:
                # 合并命中
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1

        # 从全局统计中扣除该词原有相邻 pair 计数
        for i in range(len(ids) - 1):
            pair_counter[Pair(ids[i], ids[i+1])] -= 1

        # 词重写
        tokens[wi] = out

        # 更新该词的新 pair 计数
        for i in range(len(out) - 1):
            pair_counter[Pair(out[i], out[i+1])] += 1

    # 受影响 pair 的位置表：简单起见重建（仅对 touched_words）
    # （更极致可做精细 diff，这里强调正确性+可读性）
    # 先清空所有与 touched_words 相关的 pair->pos
    affected_pairs = set()
    for wi in touched_words:
        ids = tokens[wi]
        for i in range(max(len(ids) - 1, 0)):
            affected_pairs.add(Pair(ids[i], ids[i+1]))

    # 清理：把 pair 的位置列表中过时的项去掉（粗暴做法：全量重建 positions[pi]）
    # 为了避免 O(P) 全局重建，只对受影响词重建：
    for p in list(positions.keys()):
        # 过滤掉 touched_words 的位置
        positions[p] = [(wi, i) for (wi, i) in positions[p] if wi not in touched_words]
        if not positions[p]:
            positions.pop(p, None)

    # 为受影响词重新填充位置
    for wi in touched_words:
        ids = tokens[wi]
        for i in range(len(ids) - 1):
            p = Pair(ids[i], ids[i+1])
            positions.setdefault(p, []).append((wi, i))

    # 合并掉的 pair 的位置不再需要
    positions.pop(pair, None)

# ---------------------------
# 训练主循环
# ---------------------------

def train_bpe(
    path: str,
    target_vocab_size: int,
    special_tokens: List[bytes] | None = None,
    num_workers: int = max(os.cpu_count() or 4, 4),
):
    """
    返回：
      vocab: Dict[int, bytes]  （id -> token bytes）
      merges: List[Tuple[int,int]] （按次序的 merge 规则，元素为被合并的左右 id）
    """
    # 1) 初始：byte-level 词表
    vocab = {i: bytes([i]) for i in range(BASE_VOCAB_SIZE)}
    next_id = BASE_VOCAB_SIZE

    # 2) 特殊 token（可选）
    if special_tokens:
        for st in special_tokens:
            vocab[next_id] = st
            next_id += 1

    # 3) 读取 + 并行预分词 + 初次 pair 统计
    tokens, pair_counter = initial_parallel_tokenize_and_count(path, num_workers=num_workers)

    # 4) 构建 pair->positions（后续局部更新依赖）
    positions = build_pair_positions(tokens)

    # 5) 构建最大堆（freq 取负）
    heap = [(-cnt, p) for p, cnt in pair_counter.items() if cnt > 0]
    heapq.heapify(heap)

    merges: List[Tuple[int, int]] = []
    target = target_vocab_size

    def push_if_fresh(p: Pair):
        c = pair_counter.get(p, 0)
        if c > 0:
            heapq.heappush(heap, (-c, p))

    # 6) 迭代合并
    while next_id < target and heap:
        neg_cnt, p = heapq.heappop(heap)
        cnt = -neg_cnt

        # 堆顶可能过期（旧值），用当前计数核对
        if pair_counter.get(p, 0) != cnt or cnt <= 0:
            # 过期项，跳过
            continue

        # 注册新 token
        vocab[next_id] = vocab[p.a] + vocab[p.b]
        merges.append((p.a, p.b))

        # 进行局部合并与统计更新
        apply_merge(tokens, p, next_id, pair_counter, positions)

        # 新产生的邻域 pair 计数已更新；把它们（以及可能变更的）压回堆
        # 简化：把所有可能受影响的 pair 再次压堆（堆会去重 via 过期检测）
        # 这里选取 positions 的 keys（可进一步仅推入 touched 邻域）
        for q in positions.keys():
            push_if_fresh(q)

        next_id += 1

    return vocab, merges

# ---------------------------
# 简单命令行入口
# ---------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to large text corpus")
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 4, 4))
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = ap.parse_args()

    special = [s.encode("utf-8") for s in args.special] if args.special else None
    vocab, merges = train_bpe(
        args.input,
        target_vocab_size=args.vocab_size,
        special_tokens=special,
        num_workers=args.workers,
    )

    print(f"Built vocab size: {len(vocab)}; merges: {len(merges)}")
    # 你可在此处将 vocab/merges 写入文件（JSON / merges.txt）

if __name__ == "__main__":
    main()
