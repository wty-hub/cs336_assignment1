import os
import inspect
from typing import IO, BinaryIO
import torch

from cs336_basics.transformer.optimizer import AdamW
from cs336_basics.transformer.transformer_lm import TransformerLM


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    hyperparams: dict | None = None,
):
    model_weights = model.state_dict()
    optimizer_states = optimizer.state_dict()
    checkpoint = {
        "model_weights": model_weights,
        "optimizer_states": optimizer_states,
        "iteration": iteration,
    }
    if hyperparams is not None:
        checkpoint["hyperparams"] = hyperparams
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


def _filter_init_kwargs(cls: type, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def load_checkpoint_with_hyperparams(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    device: torch.device | str | None = None,
):
    checkpoint = torch.load(src, map_location=device)
    hyperparams = checkpoint.get("hyperparams")
    if hyperparams is None:
        raise ValueError("hyperparams missing from checkpoint")

    model_kwargs = _filter_init_kwargs(TransformerLM, hyperparams)
    model = TransformerLM(**model_kwargs)
    if device is not None:
        model = model.to(device)

    model.load_state_dict(checkpoint["model_weights"])

    optimizer = None
    opt_sig = inspect.signature(AdamW.__init__)
    opt_params = set(opt_sig.parameters.keys()) - {"self", "params"}
    opt_kwargs = {}
    if "lr" in hyperparams and "lr" in opt_params:
        opt_kwargs["lr"] = hyperparams["lr"]
    if "weight_decay" in hyperparams and "weight_decay" in opt_params:
        opt_kwargs["weight_decay"] = hyperparams["weight_decay"]
    if "betas" in opt_params and "beta1" in hyperparams and "beta2" in hyperparams:
        opt_kwargs["betas"] = (hyperparams["beta1"], hyperparams["beta2"])
    optimizer = AdamW(model.parameters(), **opt_kwargs)
    if "optimizer_states" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_states"])

    # iteration = checkpoint["iteration"]
    return model, optimizer
