"""entrypoint to train and export a byte-level BPE tokenizer."""

import argparse
import json
from pathlib import Path

from cs336_basics.bpe.train_bpe import optimized_train_bpe
from tests.common import gpt2_bytes_to_unicode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer and export merges/vocab files",
    )
    parser.add_argument(
        "--input_corpus",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10_000,
    )
    return parser.parse_args()


def bytes_to_unicode_token(token_bytes: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in token_bytes)


def save_vocab(
    vocab: dict[int, bytes], save_path: Path, byte_encoder: dict[int, str]
) -> None:
    serialized_vocab = {
        bytes_to_unicode_token(token_bytes, byte_encoder): idx
        for idx, token_bytes in sorted(vocab.items(), key=lambda item: item[0])
    }
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(serialized_vocab, f, ensure_ascii=False)


def save_merges(
    merges: list[tuple[bytes, bytes]],
    save_path: Path,
    byte_encoder: dict[int, str],
):
    with save_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_token = bytes_to_unicode_token(left, byte_encoder)
            right_token = bytes_to_unicode_token(right, byte_encoder)
            f.write(f"{left_token} {right_token}\n")


def main():
    args = parse_args()
    input_corpus = Path(args.input_corpus)

    vocab, merges = optimized_train_bpe(
        input_path=input_corpus,
        vocab_size=args.vocab_size,
        special_tokens=list(args.special_tokens),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    byte_encoder = gpt2_bytes_to_unicode()
    save_vocab(vocab, save_dir / "vocab.json", byte_encoder)
    save_merges(merges, save_dir / "merges.txt", byte_encoder)

    print(f"Saved vocab to {save_dir / 'vocab.json'}")
    print(f"Saved merges to {save_dir / 'merges.txt'}")


if __name__ == "__main__":
    main()
