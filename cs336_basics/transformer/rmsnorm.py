from torch import nn
import torch


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        - d_model: int Hidden dimension of the model
        - eps: float = 1e-5 Epsilon value for numerical stability
        - device: torch.device | None = None Device to store the parameters on
        - dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x 元素平方之后，在最后一维（也就是特征维）取平均
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 保持维度，所以现在rms.shape == (batch_size, seq_length, 1)
        # 利用广播机制完成运算
        rms_norm = (x / rms) * self.g
        return rms_norm.to(in_dtype)
