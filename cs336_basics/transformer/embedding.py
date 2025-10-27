from torch import nn
import torch


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        - vocab_size: int Size of the vocabulary
        - embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        - device: torch.device | None = None Device to store the parameters on
        - dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(vocab_size, embedding_dim, device=device, dtype=dtype)
        )
        # 按照规定，截断正态初始化
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
