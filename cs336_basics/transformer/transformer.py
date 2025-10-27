from torch import nn
import torch

from cs336_basics.transformer.attention import MultiHeadSelfAttention
from cs336_basics.transformer.positionwise_feedforward import SwiGLUFFW
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.rotary_positional_embedding import (
    RotaryPositionalEmbedding,
)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        - d_model: int Dimensionality of the Transformer block inputs.
        - num_heads: int Number of heads to use in multi-head self-attention.
        - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        # 因果多头自注意力
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        # 位置前馈网络
        self.ffn = SwiGLUFFW(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        t = self.ln1(x)
        if rope is not None:
            t = self.attn(t, rope, token_positions)
        else:
            t = self.attn(t)
        x = x + t
        t = self.ln2(x)
        t = self.ln2(x)
        return x + t
