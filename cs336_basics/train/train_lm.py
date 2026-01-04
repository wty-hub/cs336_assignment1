"""Training script for the CS336 Transformer language model."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from cs336_basics.transformer.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.transformer.cross_entropy import cross_entropy
from cs336_basics.transformer.data_loading import data_loading
from cs336_basics.transformer.gradient_clipping import gradient_clipping
from cs336_basics.transformer.learning_rate_schedule import learning_rate_schedule
from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.transformer_lm import TransformerLM


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM for CS336 HW1")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)

    # Optimization hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    ## total tokens processed 327,680,000 (your batch size × total step count × context length should equal roughly this value).
    parser.add_argument("--max_steps", type=int, default=80_000)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # AdamW specific params
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # Logging / evaluation cadence
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1_000)
    parser.add_argument("--eval_iters", type=int, default=100)

    # Memory-mapped dataset paths
    parser.add_argument("--train_bin", type=str, default="data/train.bin")
    parser.add_argument("--val_bin", type=str, default="data/val.bin")
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=("uint16", "int32"),
        help="dtype for the *.bin shards",
    )

    # Checkpointing
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent checkpoint in out_dir",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1_000,
        help="Save a checkpoint every N steps",
    )

    # Weights & Biases (wandb) params
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable logging to Weights & Biases",
    )
    parser.add_argument("--wandb_project", type=str, default="cs336-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Misc (no need to modify)
    parser.add_argument("--seed", type=int, default=1_234)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def load_memmap(path: str, dtype: str) -> np.memmap:
    """使用memmap功能，不需要将整个文件读入内存，而是按需加载"""
    dtype_map = {"uint16": np.uint16, "int32": np.int32}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype}, choose from {list(dtype_map)}")

    bin_path = Path(path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing dataset shard: {bin_path}")

    return np.memmap(bin_path, dtype=dtype_map[dtype], mode="r")


def latest_checkpoint_path(out_dir: Path) -> Path | None:
    """获取最后的ckpt路径"""
    # 这里可以按照文件名排序，是因为我们将序号填充到7位
    checkpoints = sorted(out_dir.glob("ckpt_step_*.pt"))
    return checkpoints[-1] if checkpoints else None


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.memmap,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """评估过程，禁用梯度"""
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
        loss = cross_entropy(logits.view(-1, logits.size(-1)), yb.reshape(-1).long())
        losses.append(loss.item())

    model.train()
    return float(sum(losses) / len(losses))


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """没有显式地给出调整lr的接口，所以这样"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
        raise ValueError("Datasets must be longer than context_length + 1")

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
            print(f"[CKPT] Resumed from {ckpt_path} (iteration = {start_step})")
        else:
            print("[CKPT] No checkpoint found, starting fresh")

    run = maybe_init_wandb(args)
    tokens_per_batch = args.batch_size * args.context_length
    best_val = math.inf
    log_window_start = time.time()

    # 记录 losses，手动画 loss 图
    train_losses = []
    val_losses = []
    val_steps = []

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
        loss = cross_entropy(logits.view(-1, logits.size(-1)), yb.reshape(-1).long())
        train_losses.append(loss.item())
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
            
            val_losses.append(val_loss)
            val_steps.append(step + 1)

            # 画loss图
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label="Train Loss", alpha=0.3)
            plt.plot(val_steps, val_losses, label="Val Loss", marker="o")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title(f"Training Progress (Step {step+1})")
            plt.legend()
            plt.grid(True)
            plt.savefig(out_dir / "loss_curve.png")
            plt.close()

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
            print(f"[CKPT] Saved to {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
