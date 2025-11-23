from torch import Tensor
from jaxtyping import Bool, Float, Int

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
):
    """
    - inputs 就是logits
    - targets 是标签
    """
    # 先来对 logits 减去 max
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted_logits: Tensor = inputs - max_logits
    # 然后对 logits 求 exp
    logits_exp = shifted_logits.exp()
    # 求所有 logits 的指数和的对数，公式的第二项
    log_exp_sum = logits_exp.sum(dim=-1).log()
    # 使用 gather，选择出 targets 的每个值所对应的 logit
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # 正如公式，-target_logits 直接加上 log_exp_sum
    loss = -target_logits + log_exp_sum
    # 返回平均值
    return loss.mean()