"""Training loop for the future-frame UNet predictor.

Trained on normal-only UCSD training clips. Augmentation is applied
identically to the input stack and target so temporal coherence is
preserved (random horizontal flip, small spatial shift, small brightness
jitter — never rotation or per-frame independent jitter).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import UCSDPredictionDataset, VARIANTS, split_train_clips
from .losses import PredictionLoss
from .model import UNetPredictor, count_params


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    val_frac: float = 0.15
    early_stop_patience: int = 10
    seed: int = 0
    augment: bool = True
    lambda_grad: float = 1.0
    base_channels: int = 32
    window: int = 4
    warmup_epochs: int = 3


def _augment(stack: torch.Tensor, target: torch.Tensor, max_shift: int = 8
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply same random flip + shift + brightness to stack and target."""
    if torch.rand(()) < 0.5:
        stack = torch.flip(stack, dims=(-1,))
        target = torch.flip(target, dims=(-1,))
    if max_shift > 0:
        sh = int(torch.randint(-max_shift, max_shift + 1, ()))
        sw = int(torch.randint(-max_shift, max_shift + 1, ()))
        if sh or sw:
            stack = torch.roll(stack, shifts=(sh, sw), dims=(-2, -1))
            target = torch.roll(target, shifts=(sh, sw), dims=(-2, -1))
    if torch.rand(()) < 0.5:
        delta = (torch.rand(()) * 0.1 - 0.05).item()
        stack = (stack + delta).clamp(0, 1)
        target = (target + delta).clamp(0, 1)
    return stack, target


def _cosine_lr(epoch: int, total: int, warmup: int, base_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, progress)))


def train_predictor(
    dataset_root: str | Path,
    cache_root: str | Path,
    ped: str,
    variant: str,
    checkpoint_path: str | Path,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train the UNet predictor for one (ped, variant)."""
    cfg = cfg or TrainConfig()
    device = device or pick_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    spec = VARIANTS[variant]
    full = UCSDPredictionDataset(dataset_root, cache_root, ped, "Train",
                                 variant=variant, window=cfg.window)
    train_names, val_names = split_train_clips(full.clip_names, cfg.val_frac, seed=cfg.seed)

    train_ds = UCSDPredictionDataset(dataset_root, cache_root, ped, "Train",
                                     variant=variant, window=cfg.window,
                                     clip_filter=train_names, return_label=False)
    val_ds = UCSDPredictionDataset(dataset_root, cache_root, ped, "Train",
                                   variant=variant, window=cfg.window,
                                   clip_filter=val_names, return_label=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
        drop_last=True, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
        pin_memory=pin_memory,
    )

    model = UNetPredictor(in_channels=full.in_channels,
                          out_channels=full.out_channels,
                          base=cfg.base_channels).to(device)
    if verbose:
        print(f"[train] {ped}/{variant}: device={device} params={count_params(model):,}  "
              f"in_ch={full.in_channels} out_ch={full.out_channels}")
        print(f"[train] train clips={len(train_names)} ({len(train_ds)} samples)  "
              f"val clips={len(val_names)} ({len(val_ds)} samples)")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = PredictionLoss(lambda_grad=cfg.lambda_grad)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        lr = _cosine_lr(epoch, cfg.epochs, cfg.warmup_epochs, cfg.lr)
        for g in optim.param_groups:
            g["lr"] = lr

        model.train()
        t0 = time.perf_counter()
        train_loss_sum = 0.0
        n_train = 0
        for stack, target in train_loader:
            stack = stack.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if cfg.augment:
                stack, target = _augment(stack, target)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(stack)
                loss = crit(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_loss_sum += float(loss.detach()) * stack.size(0)
            n_train += stack.size(0)
        train_loss = train_loss_sum / max(n_train, 1)

        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            for stack, target in val_loader:
                stack = stack.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(stack)
                val_loss_sum += float(crit(pred, target)) * stack.size(0)
                n_val += stack.size(0)
        val_loss = val_loss_sum / max(n_val, 1)

        dt = time.perf_counter() - t0
        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss,
                        "lr": lr, "time_s": dt})
        if verbose:
            print(f"[ep {epoch + 1:03d}] train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={lr:.2e}  ({dt:.1f}s)")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(),
                        "in_channels": full.in_channels,
                        "out_channels": full.out_channels,
                        "base_channels": cfg.base_channels,
                        "window": cfg.window,
                        "ped": ped, "variant": variant},
                       checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                if verbose:
                    print(f"[train] early stop at epoch {epoch + 1} (best val {best_val:.5f})")
                break

    return {"best_val_loss": best_val, "epochs_run": len(history), "history": history}


def load_checkpoint(checkpoint_path: str | Path, device: torch.device | None = None
                    ) -> UNetPredictor:
    device = device or pick_device()
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNetPredictor(in_channels=blob["in_channels"],
                          out_channels=blob["out_channels"],
                          base=blob["base_channels"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
