from torch import nn
import torch

from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.rotary_positional_embedding import (
    RotaryPositionalEmbedding,
)
from cs336_basics.transformer.softmax import softmax
from cs336_basics.transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.config = dict(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
        )

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x, self.rope)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
