"""Frame-level anomaly evaluation for the future-frame predictor.

Per-frame score = mean L1 prediction error. Two SOTA-standard tricks are
applied before AUC computation:

1. Per-clip min-max normalization of scores. This removes inter-clip score
   bias (some clips are intrinsically harder to predict than others) and
   matches the "regularity score" convention of Hasan et al. 2016.
2. 1-D Gaussian temporal smoothing (sigma=2 frames). Anomalies are
   contiguous; isolated high-score frames are usually noise.

For frames `t < window` (no input history available), we carry forward the
first valid score so the per-clip score vector aligns with the GT vector.

We report both:
- `auc_global`:    AUC computed on concatenated scores/labels.
- `auc_per_clip`:  mean of per-clip AUCs (the metric most papers quote).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from .dataset import UCSDPredictionDataset


def equal_error_rate(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


@torch.no_grad()
def score_clip(
    model: torch.nn.Module,
    dataset: UCSDPredictionDataset,
    clip_idx: int,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) of length T_clip for one clip.

    The scores at indices [0, window) are filled by carrying forward the
    first valid score so the array aligns with the full GT vector.
    """
    T = dataset.counts[clip_idx]
    window = dataset.window
    # Local indices into dataset._index that match this clip.
    local = [(i, t) for i, (ci, t) in enumerate(dataset._index) if ci == clip_idx]

    raw_scores = np.zeros(len(local), dtype=np.float32)
    model.eval()
    for start in range(0, len(local), batch_size):
        batch = local[start:start + batch_size]
        stacks = []
        targets = []
        for di, _t in batch:
            item = dataset[di]
            if dataset.return_label:
                s, t_arr, _ = item
            else:
                s, t_arr = item
            stacks.append(s)
            targets.append(t_arr)
        x = torch.stack(stacks).to(device, non_blocking=True)
        y = torch.stack(targets).to(device, non_blocking=True)
        pred = model(x)
        err = (pred - y).abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
        for k, e in enumerate(err):
            raw_scores[start + k] = float(e)

    # Aligned to full clip length: fill [0, window) by carry-forward.
    full = np.empty(T, dtype=np.float32)
    full[window:] = raw_scores
    full[:window] = raw_scores[0] if len(raw_scores) > 0 else 0.0
    labels = dataset.labels_per_clip[clip_idx].astype(bool)
    return full, labels


@torch.no_grad()
def pixel_residual(
    model: torch.nn.Module,
    dataset: UCSDPredictionDataset,
    clip_idx: int,
    frame_idx: int,
    device: torch.device,
) -> np.ndarray:
    """Per-pixel L1 prediction error on a single (clip, target-frame).

    Returns a (H, W) float32 array. ``frame_idx`` is the target frame index
    (must be ≥ ``dataset.window`` since the model needs ``window`` past frames).
    """
    if frame_idx < dataset.window:
        raise ValueError(
            f"frame_idx={frame_idx} < window={dataset.window}; "
            "no input history available for this frame."
        )
    matches = [i for i, (ci, t) in enumerate(dataset._index)
               if ci == clip_idx and t == frame_idx]
    if not matches:
        raise IndexError(f"no dataset index for clip_idx={clip_idx}, frame_idx={frame_idx}")
    item = dataset[matches[0]]
    stack, target = item[0], item[1]
    x = stack.unsqueeze(0).to(device, non_blocking=True)
    y = target.unsqueeze(0).to(device, non_blocking=True)
    model.eval()
    pred = model(x)
    err = (pred - y).abs().squeeze(0).mean(dim=0).detach().cpu().numpy()
    return err.astype(np.float32)


def _normalize_and_smooth(scores: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smoothed = gaussian_filter1d(scores.astype(np.float32), sigma=sigma)
    s_min, s_max = float(smoothed.min()), float(smoothed.max())
    if s_max - s_min < 1e-9:
        return np.zeros_like(smoothed)
    return (smoothed - s_min) / (s_max - s_min)


def evaluate_run(
    model: torch.nn.Module,
    dataset_root: str | Path,
    cache_root: str | Path,
    ped: str,
    variant: str,
    device: torch.device,
    window: int = 4,
    batch_size: int = 16,
    smooth_sigma: float = 2.0,
) -> dict:
    """Evaluate frame-level anomaly detection on (ped, variant) test split."""
    test_ds = UCSDPredictionDataset(dataset_root, cache_root, ped, "Test",
                                    variant=variant, window=window,
                                    return_label=True)

    per_clip_scores: list[np.ndarray] = []
    per_clip_labels: list[np.ndarray] = []
    per_clip_aucs: list[float] = []

    for ci in range(len(test_ds.clip_names)):
        scores, labels = score_clip(model, test_ds, ci, device, batch_size=batch_size)
        scores_norm = _normalize_and_smooth(scores, sigma=smooth_sigma)
        per_clip_scores.append(scores_norm)
        per_clip_labels.append(labels)
        if labels.any() and (~labels).any():
            per_clip_aucs.append(float(roc_auc_score(labels, scores_norm)))
        else:
            per_clip_aucs.append(float("nan"))

    all_scores = np.concatenate(per_clip_scores)
    all_labels = np.concatenate(per_clip_labels)

    auc_global = float(roc_auc_score(all_labels, all_scores))
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    eer = equal_error_rate(fpr, tpr)
    auc_per_clip = float(np.nanmean(per_clip_aucs))

    return {
        "ped": ped,
        "variant": variant,
        "n_frames": int(all_labels.size),
        "frac_anom": float(all_labels.mean()),
        "auc_global": auc_global,
        "auc_per_clip": auc_per_clip,
        "eer": eer,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "per_clip_aucs": per_clip_aucs,
        "per_clip_scores": [s.tolist() for s in per_clip_scores],
        "per_clip_labels": [l.tolist() for l in per_clip_labels],
    }


def append_results_csv(path: str | Path, row: dict) -> None:
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ped", "variant", "n_frames", "frac_anom",
              "auc_global", "auc_per_clip", "eer"]
    write_header = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})
