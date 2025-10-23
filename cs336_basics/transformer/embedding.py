from torch import nn
import torch


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        - num_embeddings: int Size of the vocabulary
        - embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        - device: torch.device | None = None Device to store the parameters on
        - dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.W = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # 按照规定，截断正态初始化
        nn.init.trunc_normal_(self.W, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor):
        return self.W[token_ids]
