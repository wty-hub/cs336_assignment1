import math
from torch import nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.rotary_positional_embedding import (
    RotaryPositionalEmbedding,
)
from cs336_basics.transformer.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scale = torch.tensor(d_k, dtype=Q.dtype, device=Q.device).sqrt()
    # attention scores, 就是准备进行softmax的部分
    attn_scores = (Q @ K.transpose(-1, -2)) / scale  # (..., queries, keys)

    mask_bool = None
    if mask is not None:
        mask_bool = mask.to(dtype=torch.bool)
        fill_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~mask_bool, fill_value)

    attn_weights = softmax(attn_scores, dim=-1)

    if mask_bool is not None:
        attn_weights = attn_weights.masked_fill(~mask_bool, 0.0)
        weight_sums = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = torch.where(
            weight_sums == 0, torch.zeros_like(attn_weights), attn_weights / weight_sums
        )

    res = attn_weights @ V  # (..., queries, d_v)
    return res


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.ln_out = Linear(d_model, d_model)

    def forward(
        self,
        X: torch.Tensor,
        RoPE: RotaryPositionalEmbedding | None = None,
        token_position: Int[Tensor, " ... sequence_length"] | None = None,
        mask: torch.Tensor | None = None,
    ):
        """RoPE如果不为0，则说明使用旋转位置编码应用于Q，K"""
        batch_size = X.shape[0]
        seq_length = X.shape[1]
        Q = self.W_q(X).view(batch_size, seq_length, self.num_heads, self.d_k)
        K = self.W_k(X).view(batch_size, seq_length, self.num_heads, self.d_k)
        V = self.W_v(X).view(batch_size, seq_length, self.num_heads, self.d_k)

        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

        if RoPE is not None:
            # 进行 RoPE 编码
            ## 需要使得 token_position 形状与 Q，K 一致
            if token_position.dim() == 2:
                token_position = token_position.unsqueeze(1).expand(
                    batch_size, self.num_heads, seq_length
                )
            elif token_position.dim() == 3 and token_position.shape[1] == 1:
                token_position = token_position.expand(
                    batch_size, self.num_heads, seq_length
                )

            pos = token_position.reshape(batch_size * self.num_heads, seq_length)
            Q = Q.reshape(batch_size * self.num_heads, seq_length, self.d_k)
            K = K.reshape(batch_size * self.num_heads, seq_length, self.d_k)

            Q = RoPE(Q, pos)
            K = RoPE(K, pos)

            Q = Q.view(batch_size, self.num_heads, seq_length, self.d_k)
            K = K.view(batch_size, self.num_heads, seq_length, self.d_k)

        device = X.device

        # 构建因果遮罩
        causal_mask = torch.tril(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
        )
        attn_mask: torch.Tensor
        if mask is None:
            attn_mask = causal_mask
        else:
            user_mask = mask.to(dtype=torch.bool, device=device)
            attn_mask = user_mask & causal_mask

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

        res = scaled_dot_product_attention(Q, K, V, attn_mask)

        res = (
            res.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_length, self.num_heads * self.d_k)
        )

        res = self.ln_out(res)
        return res
