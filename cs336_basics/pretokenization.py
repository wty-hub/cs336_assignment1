from concurrent.futures import ProcessPoolExecutor
import mmap
import os
import sys
import time
from typing import BinaryIO
import regex
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory


def find_chunk_boundaries(
    path: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"
    file = open(path, "rb")
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    file.close()
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def delete_special_tokens(byte_text: bytes, special_tokens: list[str]):
    """删除掉 byte_text 中所有 special_tokens"""
    special_bytes = [s.encode("utf-8") for s in special_tokens]
    pattern = b"|".join(regex.escape(s) for s in special_bytes)
    return regex.sub(pattern, b"", byte_text)


# 经典的 GPT-2 分词正则
PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 编译以提升复用效率
CPD_PAT = regex.compile(PAT)


def pretokenizer_iter(text: bytes):
    """预分词使用的迭代器"""
    for s in CPD_PAT.finditer(text):
        yield s.group(0)


SPECIAL_TOKENS = None
PATH = None

def init_workers(special_tokens: list[str], path: str):
    global SPECIAL_TOKENS, PATH
    SPECIAL_TOKENS = special_tokens
    PATH = path


def run_worker_pretokenize(start: int, end: int):
    """每个进程的工作函数"""
    with open(PATH, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        chunk = memoryview(mm)[start:end].tobytes()
        if SPECIAL_TOKENS is not None and len(SPECIAL_TOKENS) > 0:
            chunk = delete_special_tokens(chunk, SPECIAL_TOKENS)
        return list(pretokenizer_iter(chunk))


def pretokenize_iter(
    path: str,
    special_tokens: list[str],
    split_special_token: bytes,
    num_processes: int,
):
    """预分词，将 file 转化为 pretoken 的序列"""
    # AI 给出的建议，每个进程分 3 个 chunk
    desired_num_chunks = num_processes * 3
    chunk_boundaries = find_chunk_boundaries(
        path, desired_num_chunks, split_special_token
    )
    # 分配任务参数
    starts, ends = [], []
    for i in range(len(chunk_boundaries) - 1):
        starts.append(chunk_boundaries[i])
        ends.append(chunk_boundaries[i + 1])
    # 使用进程池分配执行任务, 初始化全局变量
    with ProcessPoolExecutor(
        max_workers=num_processes, initializer=init_workers, initargs=(special_tokens, path)
    ) as ex:
        for ptoks in ex.map(run_worker_pretokenize, starts, ends):
            for t in ptoks:
                yield t


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-train.txt"
    pretokens = pretokenize_iter(path, [], b"<|endoftext|>", 1)
    cnt = 0
    for i in pretokens:
        cnt += 1
    print(cnt)
