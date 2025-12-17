from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict
import yaml

from cs336_basics.transformer.transformer_lm import TransformerLM


# just a helper function
def now_str():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


class Logger:
    def __init__(self, log_dir, exp_name, seed, lm: TransformerLM, flushing_steps=20):
        """flushing_steps 就是多少个 step 刷新一下缓冲区"""
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.config = dict()
        # 时间原点（开始记录的时间）
        self.t0 = time.time()

        self.config["exp_name"] = exp_name
        self.config["created_at"] = self.t0
        self.config["lm_config"] = lm.config
        self.config["seed"] = seed

        self.flushing_steps = flushing_steps

        self.log_path: Path = Path(self.log_dir) / self.exp_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # 写入log的计数
        self.write_count = 0
        # 每行一个json
        self.log_fp = open(self.log_path / "metrics.jsonl", "w")
        self.log_event("meta", -1, {}, {"event": "start", "time": now_str()})

    def save_config(self):
        config_file_path = self.log_path / "config.yaml"
        with open(config_file_path, "w") as f:
            yaml.safe_dump(self.config, f)

    @property
    def elapsed(self):
        return time.time() - self.t0
    
    def log_event(self, split, step, metrics, extra=None):
        record = {
            "time": now_str(),
            "elapsed_sec": round(self.elapsed, 6),
            "step": step,
            "split": split,
            "metrics": metrics,
            "extra": extra or {},
        }
        self.log_fp.write(json.dumps(record) + "\n")
        self.write_count += 1
        if self.write_count % self.flushing_steps == 0:
            self.log_fp.flush()

    def log_train(self, step, **metrics):
        self.log_event("train", step, metrics)

    def log_val(self, step, **metrics):
        self.log_event("val", step, metrics)

    def close(self):
        self.log_event(
            "meta",
            -1,
            metrics={},
            extra={"event": "end", "time": now_str()},
        )
        self.log_fp.close()

    def __del__(self):
        self.close()



def plot_curves(
    log_dir: str | Path,
    metric_key: str = "loss",
    out_png: str = "curve.png",
) -> Path:
    """
    Reads metrics.jsonl and plots:
    - metric vs step  (train + val)
    - metric vs elapsed_sec (train + val)
    Saves a single png with two plots stacked vertically.
    """
    import matplotlib.pyplot as plt

    log_dir = Path(log_dir)
    path = log_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    train_step, train_t, train_y = [], [], []
    val_step, val_t, val_y = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            split = rec.get("split")
            m = rec.get("metrics", {})
            if metric_key not in m:
                continue
            y = m[metric_key]
            step = rec.get("step", 0)
            t = rec.get("elapsed_sec", 0.0)
            if split == "train":
                train_step.append(step)
                train_t.append(t)
                train_y.append(y)
            elif split == "val":
                val_step.append(step)
                val_t.append(t)
                val_y.append(y)

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    if train_step:
        ax1.plot(train_step, train_y, label=f"train/{metric_key}")
    if val_step:
        ax1.plot(val_step, val_y, label=f"val/{metric_key}")
    ax1.set_xlabel("step")
    ax1.set_ylabel(metric_key)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    if train_t:
        ax2.plot(train_t, train_y, label=f"train/{metric_key}")
    if val_t:
        ax2.plot(val_t, val_y, label=f"val/{metric_key}")
    ax2.set_xlabel("wallclock (sec)")
    ax2.set_ylabel(metric_key)
    ax2.legend()

    out_path = log_dir / out_png
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path