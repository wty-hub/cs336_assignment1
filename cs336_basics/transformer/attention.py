from torch import nn
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.transformer.softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    attn_weights = softmax(Q @ K.transpose(-1, -2) / torch.sqrt(d_k), -1)   # (..., queries, keys)
    


