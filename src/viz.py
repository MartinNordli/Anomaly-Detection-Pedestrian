"""Qualitative visualization helpers: L/S decompositions and ROC overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cache import load_cached


def visualize_decomposition(
    cache_root: str | Path,
    ped: str,
    split: str,
    clip_name: str,
    frame_indices: list[int] | None = None,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Render a 3-row figure: raw / L / |S| for the chosen frames of one clip."""
    blob = load_cached(cache_root, ped, split, clip_name)
    X = blob["X"][:].astype(np.float32) / 255.0  # (T,H,W)
    L = blob["L"][:].astype(np.float32)
    S = blob["S"][:].astype(np.float32)
    T = X.shape[0]

    if frame_indices is None:
        frame_indices = list(np.linspace(0, T - 1, 6).astype(int))

    n = len(frame_indices)
    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 5))
    if n == 1:
        axes = axes.reshape(3, 1)

    s_abs = np.abs(S)
    s_vmax = float(np.percentile(s_abs, 99.5))

    for j, fi in enumerate(frame_indices):
        axes[0, j].imshow(X[fi], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"frame {fi}", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(L[fi], cmap="gray")
        axes[1, j].axis("off")
        axes[2, j].imshow(s_abs[fi], cmap="hot", vmin=0, vmax=s_vmax)
        axes[2, j].axis("off")
    axes[0, 0].set_ylabel("X (raw)", fontsize=10)
    axes[1, 0].set_ylabel("L", fontsize=10)
    axes[2, 0].set_ylabel("|S|", fontsize=10)
    fig.suptitle(f"{ped}/{split}/{clip_name}", fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_roc_overlay(
    runs: list[dict],
    title: str,
    save_path: str | Path | None = None,
    score_kind: str = "mean",
) -> "plt.Figure":
    """Overlay ROC curves for the four representations on one axes."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in runs:
        fpr, tpr = r[f"roc_{score_kind}"]
        auc = r[f"auc_{score_kind}"]
        ax.plot(fpr, tpr, label=f"{r['representation']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_score_timeline(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Plot anomaly scores over time with GT shaded — useful for failure analysis."""
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(len(scores))
    ax.plot(t, scores, color="tab:blue", lw=0.8)
    ax.fill_between(t, 0, scores.max(), where=labels, color="tab:red", alpha=0.15,
                    label="anomalous (GT)")
    ax.set_xlabel("frame")
    ax.set_ylabel("recon error")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
