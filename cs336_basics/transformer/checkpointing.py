import os
from typing import IO, BinaryIO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_weights = model.state_dict()
    optimizer_states = optimizer.state_dict()
    checkpoint = {
        "model_weights": model_weights,
        "optimizer_states": optimizer_states,
        "iteration": iteration,
    }
    torch.save(obj=checkpoint, f=out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_weights"])
    optimizer.load_state_dict(checkpoint["optimizer_states"])
    return checkpoint["iteration"]
