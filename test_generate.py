from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.transformer.checkpointing import load_checkpoint_with_hyperparams
from cs336_basics.transformer.transformer_lm import TransformerLM
from cs336_basics.util.generate_text import generate_text

model, _ = load_checkpoint_with_hyperparams(
    '/root/autodl-tmp/cs336_assignment1/checkpoints/test_generate/ckpt_step_0160000.pt')
tokenizer = Tokenizer.from_files('/root/autodl-tmp/cs336_assignment1/bpe/vocab.json',
                                 '/root/autodl-tmp/cs336_assignment1/bpe/merges.txt', special_tokens=["<|endoftext|>"])

origin_text = "Once upon a time there is a little boy"
print(
    f'generated text: {generate_text(tokenizer, model, origin_text, 1024, 0.8, "<|endoftext|>")}')
