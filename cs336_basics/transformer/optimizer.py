from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        # 超参数合法性检查
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    # 第一次使用这个参数，初始化状态
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                # 这两个名字是抄来的，显得专业
                exp_avg: torch.Tensor = state[
                    "exp_avg"
                ]  # 就是 m_t，Exponential Moving Average
                exp_avg_sq: torch.Tensor = state[
                    "exp_avg_sq"
                ]  # v_t，Exponential Moving Average of squared gradients
                # 从 1 开始
                state["step"] += 1
                t = state["step"]
                # torch 中，后面带 _ 的方法表示原地修改变量
                # 这里就是 m，v 的更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 偏差矫正项
                exp_avg_bias_correction = 1.0 - beta1**t  # m 的分母
                exp_avg_sq_correction = 1.0 - beta2**t  # v 的分母
                # 这里为了方便计算，我把 m 与其偏差矫正项拆开
                first_factor = lr / exp_avg_bias_correction
                denom = (exp_avg_sq.sqrt() / math.sqrt(exp_avg_sq_correction)) + eps
                # 权重衰减
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.addcdiv_(exp_avg, denom, value=-first_factor)


if __name__ == "__main__":
    for lr in [1e1, 1e2, 1e3]:
        print(f"lr: {lr}")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=1)
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
