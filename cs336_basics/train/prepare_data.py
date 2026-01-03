import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from cs336_basics.bpe.tokenizer import Tokenizer

def prepare(input_path, output_path, tokenizer):
    print(f"Processing {input_path} -> {output_path}")
    buffer = []
    buffer_size = 1024 * 1024 
    
    with open(output_path, "wb") as f_out:
        with open(input_path, "r", encoding="utf-8") as f_in:
            for token_id in tqdm(tokenizer.encode_iterable(f_in)):
                buffer.append(token_id)
                if len(buffer) >= buffer_size:
                    np.array(buffer, dtype=np.uint16).tofile(f_out)
                    buffer = []
        
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--val_input", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--train_output", type=str, default="data/train.bin")
    parser.add_argument("--val_output", type=str, default="data/val.bin")
    parser.add_argument("--vocab", type=str, default="bpe/vocab.json")
    parser.add_argument("--merges", type=str, default="bpe/merges.txt")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    
    if Path(args.train_input).exists():
        prepare(args.train_input, args.train_output, tokenizer)
    else:
        print(f"Warning: {args.train_input} not found.")

    if Path(args.val_input).exists():
        prepare(args.val_input, args.val_output, tokenizer)
    else:
        print(f"Warning: {args.val_input} not found.")

if __name__ == "__main__":
    main()
