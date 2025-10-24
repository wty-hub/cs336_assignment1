from torch import nn
import torch
from cs336_basics.transformer.linear import Linear


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLUFFW(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        """
        使用SwiGLU的位置前馈网络
        - d_ff: 内部升维的维度
        """
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            # 按照建议，将d_ff设置为 8/3 * d_model, 向上取整到64的倍数
            d_ff = ((int(8 / 3 * d_model) + 63) // 64) * 64
            self.d_ff = d_ff
        else:
            self.d_ff = d_ff

        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        x_1 = SiLU(self.W1(x))
        x_2 = self.W3(x)
        # 这里是逐元素相乘
        return self.W2(x_1 * x_2)
