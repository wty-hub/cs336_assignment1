from torch import nn
import torch


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Construct the RoPE module and create buffers if needed.
        - theta: float Θ value for the RoPE
        - d_k: int dimension of query and key vectors
        - max_seq_len: int Maximum sequence length that will be inputted
        - device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # positions 就是 i
        positions = torch.arange(max_seq_len)
        # 计算角度时的分母（资料中从1开始，这里从0开始）
        denominator = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        )
        ## 要让每个i除以每个分母，这样就需要使用外积
        ## 第一维是i，第二位是k
        angles = torch.outer(positions, denominator)  # (max_seq_len, d_k / 2)
        ## 重复一遍，以便forward中进行奇偶分组计算
        sin_angles = torch.sin(angles).repeat_interleave(
            2, dim=-1
        )  # (max_seq_len, d_k)
        cos_angles = torch.cos(angles).repeat_interleave(
            2, dim=-1
        )  # (max_seq_len, d_k)
        if device is not None:
            sin_angles = sin_angles.to(device)
            cos_angles = cos_angles.to(device)
        # 使用register_buffer注册为固定的参数, persistent设为False，表示可以再次生成，不用保存
        self.register_buffer("sin_angles", sin_angles, persistent=False)
        self.register_buffer("cos_angles", cos_angles, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension
        """
        # 取出相应位置的cos和sin值
        position_sins = self.sin_angles[token_positions]  # (seq_len, d_k)
        position_coss = self.cos_angles[token_positions]  # (seq_len, d_k)

        # 将x分为奇偶
        x_even = x[..., 0::2]  # (..., seq_len, d_k / 2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k / 2)
        # cos和sin对应的乘数
        cos_mult = position_coss[..., 0::2]  # (seq_len, d_k / 2)
        sin_mult = position_sins[..., 0::2]  # (seq_len, d_k / 2)
        # 利用广播机制完成旋转计算
        res_even = x_even * cos_mult - x_odd * sin_mult
        res_odd = x_even * sin_mult + x_odd * cos_mult
        res = torch.empty_like(x)
        res[..., 0::2] = res_even
        res[..., 1::2] = res_odd
        return res
