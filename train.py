import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.transformer.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.transformer.cross_entropy import cross_entropy
from cs336_basics.transformer.data_loading import data_loading
from cs336_basics.transformer.gradient_clipping import gradient_clipping
from cs336_basics.transformer.learning_rate_schedule import learning_rate_schedule
from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.transformer_lm import TransformerLM


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM for CS336 HW1")

    # 模型架构超参数
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=1600)
    parser.add_argument("--d_ff", type=int, default=6400)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 训练相关超参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # AdamW 的超参数
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # eval / log 频率
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1_000)
    parser.add_argument("--eval_iters", type=int, default=100)

    # 预存储的数据集路径（np.memmap）
    parser.add_argument("--train_bin", type=str, default="train.bin")
    parser.add_argument("--val_bin", type=str, default="val.bin")
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32"],
        help="train.bin / val.bin 的 dtype",
    )

    # ====== checkpoint / 输出路径 ======
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="如果为 True，则尝试从 out_dir 中最近的 ckpt 恢复训练",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1_000,
        help="每多少步保存一次 checkpoint",
    )

    # ====== wandb 日志 ======
    parser.add_argument(
        "--wandb", action="store_true", help="是否使用 Weights & Biases 做训练日志"
    )
    parser.add_argument("--wandb_project", type=str, default="cs336-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # ====== 设备 / 杂项 ======
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def load_memmap(path: str, dtype: str) -> np.memmap:
    dtype_map = {"uint16": np.uint16, "int32": np.int32}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype}, choose from {list(dtype_map)}")

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    return np.memmap(data_path, dtype=dtype_map[dtype], mode="r")


def latest_checkpoint_path(out_dir: Path) -> Path | None:
    checkpoints = sorted(out_dir.glob("ckpt_step_*.pt"))
    return checkpoints[-1] if checkpoints else None


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.memmap,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(args.eval_iters):
        xb, yb = data_loading(
            dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            dtype=torch.long,
        )
        logits = model(xb.long())
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.reshape(-1).long(),
        )
        losses.append(loss.item())

    model.train()
    return float(sum(losses) / len(losses))


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maybe_init_wandb(args: argparse.Namespace):
    if not args.wandb:
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        mode="online",
    )



def main() -> None:
    args = get_args()
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_memmap(args.train_bin, args.dtype)
    val_data = load_memmap(args.val_bin, args.dtype)
    min_required = args.context_length + 1
    if train_data.shape[0] <= min_required or val_data.shape[0] <= min_required:
        raise ValueError("数据集长度必须大于 context_length + 1，请检查 *.bin 文件")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        ckpt_path = latest_checkpoint_path(out_dir)
        if ckpt_path is not None:
            start_step = load_checkpoint(ckpt_path, model, optimizer)
            print(f"[CKPT] 恢复自 {ckpt_path} (iteration = {start_step})")
        else:
            print("[CKPT] 未找到 checkpoint，重新开始训练")

    run = maybe_init_wandb(args)
    tokens_per_batch = args.batch_size * args.context_length
    best_val = math.inf
    log_window_start = time.time()

    model.train()
    for step in range(start_step, args.max_steps):
        lr = learning_rate_schedule(
            it=step,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.max_steps,
        )
        set_learning_rate(optimizer, lr)

        xb, yb = data_loading(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
            dtype=torch.long,
        )
        logits = model(xb.long())
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.reshape(-1).long(),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if run is not None:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "lr": lr,
                    "tokens_per_batch": tokens_per_batch,
                },
                step=step + 1,
            )

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - log_window_start
            tok_per_sec = tokens_per_batch * args.log_interval / max(elapsed, 1e-9)
            print(
                f"step {step+1:07d} | loss {loss.item():.4f} | lr {lr:.2e} | tok/s {tok_per_sec:,.0f}"
            )
            log_window_start = time.time()

        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args, device)
            best_val = min(best_val, val_loss)
            print(
                f"[Eval] step {step+1:07d} | val_loss {val_loss:.4f} | best {best_val:.4f}"
            )
            if run is not None:
                wandb.log(
                    {"val/loss": val_loss, "val/best_loss": best_val},
                    step=step + 1,
                )

        if (step + 1) % args.checkpoint_interval == 0 or (step + 1) == args.max_steps:
            ckpt_path = out_dir / f"ckpt_step_{step+1:07d}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"[CKPT] 已保存到 {ckpt_path}")

    if run is not None:
        run.finish()

