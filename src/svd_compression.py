"""SVD-based weight compression for the trained UNet (RQ4 ablation).

For each Conv2d layer with weight tensor W of shape (out, in, kh, kw), we
reshape to a 2-D matrix M of shape (out, in*kh*kw), compute SVD, truncate to
rank `k = round(rank_frac * min(out, in*kh*kw))`, reconstruct, and write
back. This is a hard-rank-k constraint on the reshaped weight matrix and ties
directly to the singular-value-thresholding step inside the RPCA solver — so
the project uses SVD twice at conceptually distinct levels (data
preprocessing and model compression), as the proposal anticipates.

The "effective parameter count" reported is `k * (out + in*kh*kw)` summed
over all Conv2d layers — the cost of storing the truncated factors.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .evaluate import evaluate_run


def truncate_conv_weights(model: nn.Module, rank_frac: float) -> nn.Module:
    """Return a deep copy of `model` with each Conv2d weight low-rank-truncated.

    rank_frac=1.0 returns a clone unchanged.
    """
    m = copy.deepcopy(model)
    if rank_frac >= 1.0:
        return m
    with torch.no_grad():
        for module in m.modules():
            if isinstance(module, nn.Conv2d):
                W = module.weight.data
                out_c, in_c, kh, kw = W.shape
                M = W.reshape(out_c, in_c * kh * kw)
                # Apple Accelerate SVD on CPU is the most reliable path.
                M_cpu = M.detach().to("cpu", torch.float32)
                U, S, Vh = torch.linalg.svd(M_cpu, full_matrices=False)
                k = max(1, int(round(rank_frac * min(M.shape))))
                Mk = (U[:, :k] * S[:k]) @ Vh[:k]
                module.weight.data = Mk.to(W.device, W.dtype).reshape(out_c, in_c, kh, kw)
    return m


def effective_param_count(model: nn.Module, rank_frac: float) -> int:
    """Storage cost of low-rank-factored Conv2d weights (other params unchanged)."""
    n = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            out_c, in_c, kh, kw = module.weight.shape
            d = min(out_c, in_c * kh * kw)
            k = max(1, int(round(rank_frac * d)))
            n += k * (out_c + in_c * kh * kw)
            if module.bias is not None:
                n += module.bias.numel()
        else:
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    n += p.numel()
    return n


def evaluate_compression_curve(
    model: nn.Module,
    dataset_root: str | Path,
    cache_root: str | Path,
    ped: str,
    variant: str,
    device: torch.device,
    rank_fracs: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.1),
    window: int = 4,
    batch_size: int = 16,
) -> list[dict]:
    """Sweep rank_frac, evaluate each truncation, return list of summary rows."""
    full_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows: list[dict] = []
    for rf in rank_fracs:
        m_trunc = truncate_conv_weights(model, rank_frac=rf)
        eff = effective_param_count(m_trunc, rank_frac=rf)
        res = evaluate_run(m_trunc, dataset_root, cache_root, ped, variant,
                           device=device, window=window, batch_size=batch_size)
        rows.append({
            "rank_frac": float(rf),
            "effective_params": int(eff),
            "compression_ratio": float(eff) / float(full_params),
            "auc_global": res["auc_global"],
            "auc_per_clip": res["auc_per_clip"],
            "eer": res["eer"],
        })
    return rows


def append_compression_csv(path: str | Path, ped: str, variant: str, rows: list[dict]) -> None:
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ped", "variant", "rank_frac", "effective_params",
              "compression_ratio", "auc_global", "auc_per_clip", "eer"]
    write_header = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({"ped": ped, "variant": variant, **r})
