import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a linear transformation module.
        - in_features: int final dimension of the input
        - out_features: int final dimension of the output
        - device: torch.device | None = None Device to store the parameters on
        - dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # variance，方差
        var = 2.0 / (in_features + out_features)
        std = math.sqrt(var)
        # 截断初始化
        nn.init.trunc_normal_(self.W, 0, std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor):
        # x.shape: ..., in_features
        return x @ self.W.T
