import os
import regex
import time
from tqdm import tqdm
import pickle


# def find_chunk_boundaries(
#     path: str,
#     desired_num_chunks: int,
#     split_special_token: bytes,
# ) -> list[int]:
#     """
#     Chunk the file into parts that can be counted independently.
#     May return fewer chunks if the boundaries end up overlapping.
#     """
#     assert isinstance(
#         split_special_token, bytes
#     ), "Must represent special token as a bytestring"
#     file = open(path, "rb")
#     # Get total file size in bytes
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)

#     chunk_size = file_size // desired_num_chunks

#     # Initial guesses for chunk boundary locations, uniformly spaced
#     # Chunks start on previous index, don't include last index
#     chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
#     chunk_boundaries[-1] = file_size

#     mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

#     for bi in tqdm(
#         range(1, len(chunk_boundaries) - 1), desc="Finding Chunk Boundaries"
#     ):
#         initial_position = chunk_boundaries[bi]
#         file.seek(initial_position)  # Start at boundary guess
#         while True:
#             mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

#             # If EOF, this boundary should be at the end of the file
#             if mini_chunk == b"":
#                 chunk_boundaries[bi] = file_size
#                 break

#             # Find the special token in the mini chunk
#             found_at = mini_chunk.find(split_special_token)
#             if found_at != -1:
#                 chunk_boundaries[bi] = initial_position + found_at
#                 break
#             initial_position += mini_chunk_size
#     file.close()
#     # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
#     return sorted(set(chunk_boundaries))


# def delete_special_tokens(byte_text: bytes, special_tokens: list[str]):
#     """删除掉 byte_text 中所有 special_tokens"""
#     special_bytes = [s.encode("utf-8") for s in special_tokens]
#     pattern = b"|".join(regex.escape(s) for s in special_bytes)
#     return regex.sub(pattern, b"", byte_text)


# 经典的 GPT-2 分词正则
PAT = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 编译以提升复用效率
CPD_PAT = regex.compile(PAT)


def pretokenizer(text: bytes):
    """进行预分词的迭代器"""
    for s in CPD_PAT.finditer(text):
        yield s.group(0)


def single_pretokenize_iter(
    path: str,
    special_tokens: list[str],
    split_special_token: bytes,
):
    """预分词，将 file 转化为 pretoken 的序列, 返回迭代器以节省内存"""
    file = open(path, "rb")
    chunk = file.read()
    chunks = regex.split(regex.escape(split_special_token), chunk)
    pattern_to_remove = b"|".join(regex.escape(s.encode()) for s in special_tokens)
    for i in range(len(chunks)):
        chunks[i] = regex.sub(pattern_to_remove, b"", chunks[i])
    for chunk in chunks:
        for m in pretokenizer(chunk):
            yield m
    file.close()


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-train.txt"
    pretokens = single_pretokenize_iter(path, [], b"<|endoftext|>", 16)
    res = []
    i = 0
    for p in pretokens:
        i += 1
        res.append(p)

    output_path = "tmp/TinyStoriesV2-GPT4-train-pretokens.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(res, f)
    print(f"Saved pre-tokens to {output_path}")

    print(f"total {i} pre-tokens")
