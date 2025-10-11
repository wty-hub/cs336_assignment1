import os
import sys
import time
from typing import BinaryIO
import regex
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory


def find_chunk_boundaries(
    buf: memoryview,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将原函数改为使用共享内存的
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    file_size = buf.nbytes

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # 对中间的每个边界进行向后搜索，找到最接近的 split_special_token, 避免分割原有的整段文章
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        while True:
            end = min(file_size, initial_position + mini_chunk_size)
            if end == initial_position:
                chunk_boundaries[bi] = file_size
                break

            mini_chunk = bytes(buf[initial_position:end])

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position = end

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def delete_special_tokens(byte_text: bytes, special_tokens: list[str]):
    """删除掉 byte_text 中所有 special_tokens"""
    special_bytes = [s.encode("utf-8") for s in special_tokens]
    pattern = "|".join(regex.escape(s) for s in special_bytes)
    return regex.sub(pattern, "", byte_text)


# 经典的 GPT-2 分词正则
PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 编译以提升复用效率
CPD_PAT = regex.compile(PAT)


def pretokenizer(text: bytes):
    """预分词使用的迭代器"""
    for s in regex.finditer(CPD_PAT, text):
        yield s.group(0)


def pretokenize(
    path: str,
    special_tokens: list[str],
    split_special_token: bytes,
    num_processes: int,
) -> list[bytes]:
    """预分词，将 file 转化为 pretoken 的序列"""
    # 读取文件
    with open(path, "rb") as f:
        data = f.read()
    # 开共享内存
    size = len(data)
    shm = SharedMemory(create=True, size=size)
    try:
        shm.buf[:size] = data
        # AI 给出的建议，每个进程分 3 个 chunk
        desired_num_chunks = num_processes * 3
        chunk_boundaries = find_chunk_boundaries(
            shm.buf, desired_num_chunks, split_special_token
        )
        # 分配任务参数
        tasks = []
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i+1]
            tasks.append((shm.name, start, end, ))
    finally:
        # 收尾
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, "rb") as f:
        text = f.read()
        start = time.perf_counter()
        chunks = make_chunk(text, 4)
        print(chunks)
        end = time.perf_counter()
        print(f"分块用了 {end - start : .4f} 秒")
