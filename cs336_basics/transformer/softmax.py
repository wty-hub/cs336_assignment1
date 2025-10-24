import torch


def softmax(x: torch.Tensor, dim: int):
    """
    - x: 输入向量
    - dim: softmax进行的维度
    """
    # 保持维度以便广播
    x_max = x.max(dim=dim, keepdim=True)[0]
    # 减去最大值，使所有值为负，防止溢出
    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    res = x_exp / x_exp_sum
    return res
