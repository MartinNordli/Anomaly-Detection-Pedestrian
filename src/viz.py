"""Qualitative + diagnostic visualizations for the RPCA anomaly-detection report.

The plotting helpers fall into three groups:

- *Qualitative*: ``visualize_decomposition`` shows the X/L/|S| split for chosen
  frames; useful to confirm that S concentrates on moving anomalies.
- *Aggregate metrics*: ``plot_roc_overlay``, ``plot_per_clip_auc_bar``,
  ``plot_score_histograms`` summarise variant-level performance.
- *Per-frame diagnostics*: ``plot_score_timeline_overlay``,
  ``plot_pixel_error_panel``, ``plot_singular_spectrum`` and
  ``plot_rpca_convergence`` zoom into a single clip / frame / decomposition.

All helpers accept ``save_path`` (without extension) and route through
``save_figure`` to write both PNG (web/notebook) and PDF (vector for LaTeX).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .cache import load_cached
from .data import list_clips, parse_gt_m, clip_frame_counts


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def save_figure(fig: "plt.Figure", save_path_stem: str | Path | None,
                *, dpi: int = 120, formats: Sequence[str] = ("png", "pdf")) -> None:
    """Save ``fig`` to disk in multiple formats. Idempotent if path is None.

    ``save_path_stem`` may include or omit an extension; if it already ends in
    one of the requested formats the suffix is stripped before writing.
    """
    if save_path_stem is None:
        return
    p = Path(save_path_stem)
    if p.suffix.lower().lstrip(".") in {"png", "pdf", "svg", "jpg", "jpeg"}:
        p = p.with_suffix("")
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(p.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")


def _normalize_and_smooth(scores: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Per-clip min-max normalize after Gaussian smoothing (mirrors evaluate.py)."""
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(scores.astype(np.float32), sigma=sigma)
    s_min, s_max = float(smoothed.min()), float(smoothed.max())
    if s_max - s_min < 1e-9:
        return np.zeros_like(smoothed)
    return (smoothed - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# 1. Qualitative L/S decomposition
# ---------------------------------------------------------------------------

def visualize_decomposition(
    cache_root: str | Path,
    ped: str,
    split: str,
    clip_name: str,
    frame_indices: list[int] | None = None,
    save_path: str | Path | None = None,
    *,
    gt_labels: np.ndarray | None = None,
) -> "plt.Figure":
    """Render a 3-row figure: raw / L / |S| for the chosen frames of one clip.

    If ``gt_labels`` (length T bool array) is provided, the title row marks
    anomalous frames with a red "(anom)" tag — useful for the modified
    qualitative figure that includes anomalous frames.
    """
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
        is_anom = bool(gt_labels[fi]) if gt_labels is not None else False
        title_color = "tab:red" if is_anom else "black"
        title = f"frame {fi}" + (" (anom)" if is_anom else "")
        axes[0, j].imshow(X[fi], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(title, fontsize=8, color=title_color)
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
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. ROC overlay
# ---------------------------------------------------------------------------

def plot_roc_overlay(
    runs: list[dict],
    title: str,
    save_path: str | Path | None = None,
    *,
    label_key: str = "variant",
    auc_key: str = "auc_global",
) -> "plt.Figure":
    """Overlay ROC curves for any number of runs sharing the evaluate.py schema.

    Each run dict must contain ``roc_fpr`` and ``roc_tpr`` (lists or arrays)
    plus a label and AUC accessible via ``label_key`` and ``auc_key``.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in runs:
        fpr = np.asarray(r["roc_fpr"])
        tpr = np.asarray(r["roc_tpr"])
        auc = r[auc_key]
        ax.plot(fpr, tpr, label=f"{r[label_key]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Score timeline (single + overlay)
# ---------------------------------------------------------------------------

def plot_score_timeline(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Plot a single anomaly-score curve with GT shaded."""
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(len(scores))
    ax.plot(t, scores, color="tab:blue", lw=0.8)
    ax.fill_between(t, 0, float(scores.max()), where=labels.astype(bool),
                    color="tab:red", alpha=0.15, label="anomalous (GT)")
    ax.set_xlabel("frame")
    ax.set_ylabel("recon error")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


def plot_score_timeline_overlay(
    scores_by_variant: dict[str, np.ndarray],
    labels: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Overlay several variants' anomaly-score curves on one axes with GT shading.

    Each curve is plotted on its own min-max-normalised scale so visual height
    is comparable; the GT-anomalous frames are drawn as a light grey band.
    """
    fig, ax = plt.subplots(figsize=(11, 3.4))
    t = np.arange(len(labels))
    ax.fill_between(t, 0, 1, where=labels.astype(bool), color="0.85",
                    label="GT anomaly")
    colors = plt.cm.tab10.colors
    for i, (name, s) in enumerate(scores_by_variant.items()):
        s = np.asarray(s, dtype=np.float32)
        s_min, s_max = float(s.min()), float(s.max())
        s01 = (s - s_min) / (s_max - s_min) if s_max - s_min > 1e-9 else np.zeros_like(s)
        ax.plot(t, s01, color=colors[i % len(colors)], lw=1.1, label=name)
    ax.set_xlabel("frame")
    ax.set_ylabel("normalised score")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, ncols=2)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Pixel-error panel (anomaly heatmap + failure cases)
# ---------------------------------------------------------------------------

def plot_pixel_error_panel(
    panels: list[tuple[str, np.ndarray, str]],
    title: str = "",
    save_path: str | Path | None = None,
    *,
    figsize_per_panel: tuple[float, float] = (3.4, 3.0),
) -> "plt.Figure":
    """Generic side-by-side panel figure.

    ``panels`` is a list of ``(subtitle, image_2d, cmap)`` tuples. Each image
    is auto-scaled to its own ``vmin/vmax`` (5th/99.5th percentile) so that
    heatmaps with very different ranges remain readable side-by-side.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n,
                                            figsize_per_panel[1]))
    if n == 1:
        axes = [axes]
    for ax, (sub, img, cmap) in zip(axes, panels):
        finite = img[np.isfinite(img)]
        if finite.size and cmap != "gray":
            vmin, vmax = float(np.percentile(finite, 5)), float(np.percentile(finite, 99.5))
        else:
            vmin, vmax = (0.0, 1.0) if cmap == "gray" else (None, None)
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(sub, fontsize=9)
        ax.axis("off")
        if cmap != "gray":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Score histograms (normal vs anomal × variants)
# ---------------------------------------------------------------------------

def plot_score_histograms(
    runs: list[dict],
    title: str = "",
    save_path: str | Path | None = None,
    *,
    bins: int = 40,
    label_key: str = "variant",
    auc_key: str = "auc_global",
) -> "plt.Figure":
    """For each run, overlay normal-vs-anomal histograms of per-frame scores.

    Expects ``per_clip_scores`` and ``per_clip_labels`` on each run dict — both
    are populated by ``evaluate_run``. Concatenated across clips before plotting.
    """
    n = len(runs)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.2 * nrows),
                             squeeze=False)

    for ax, r in zip(axes.flat, runs):
        scores = np.concatenate([np.asarray(s) for s in r["per_clip_scores"]])
        labels = np.concatenate([np.asarray(l) for l in r["per_clip_labels"]]).astype(bool)
        ax.hist(scores[~labels], bins=bins, density=True, alpha=0.55,
                color="tab:blue", label="normal")
        ax.hist(scores[labels], bins=bins, density=True, alpha=0.55,
                color="tab:red", label="anomalous")
        ax.set_title(f"{r[label_key]} (AUC={r[auc_key]:.3f})", fontsize=10)
        ax.set_xlabel("normalised score")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes.flat[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. ‖S‖₁ baseline + boxplot
# ---------------------------------------------------------------------------

def _per_clip_s_l1(cache_root: str | Path, ped: str, split: str, clip_name: str
                   ) -> np.ndarray:
    """Return per-frame ‖S_t‖₁ for a single clip, shape (T,)."""
    blob = load_cached(cache_root, ped, split, clip_name)
    S = blob["S"][:].astype(np.float32)
    return np.abs(S).sum(axis=(1, 2))


def s_norm_baseline_run(
    dataset_root: str | Path,
    cache_root: str | Path,
    ped: str,
    *,
    split: str = "Test",
    smooth_sigma: float = 2.0,
    variant_label: str = "|S|_1",
) -> dict:
    """Evaluate ‖S‖₁ as a frame-level anomaly score on the ``ped`` test split.

    Mirrors the schema of ``evaluate_run`` so the result drops straight into
    ``plot_roc_overlay``: ``auc_global``, ``auc_per_clip``, ``eer``,
    ``roc_fpr``, ``roc_tpr``, ``per_clip_aucs``, ``per_clip_scores``,
    ``per_clip_labels`` are all populated. No model required — purely a
    diagnostic on the cached RPCA decomposition.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    from .evaluate import equal_error_rate

    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    split_dir = dataset_root / ped / split
    kind = "train" if split.lower().startswith("train") else "test"

    clip_dirs = list_clips(split_dir, kind=kind)
    clip_names = [c.name for c in clip_dirs]
    counts = clip_frame_counts(split_dir, kind=kind)
    if kind == "test":
        labels_per_clip = parse_gt_m(split_dir / f"{ped}.m", n_frames_per_clip=counts)
    else:
        labels_per_clip = [np.zeros(c, dtype=bool) for c in counts]

    per_clip_scores: list[np.ndarray] = []
    per_clip_labels: list[np.ndarray] = []
    per_clip_aucs: list[float] = []

    for name, lab in zip(clip_names, labels_per_clip):
        s = _per_clip_s_l1(cache_root, ped, split, name)
        s_norm = _normalize_and_smooth(s, sigma=smooth_sigma)
        per_clip_scores.append(s_norm)
        per_clip_labels.append(lab.astype(bool))
        if lab.any() and (~lab).any():
            per_clip_aucs.append(float(roc_auc_score(lab, s_norm)))
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
        "variant": variant_label,
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
        "clip_names": clip_names,
    }


def plot_s_norm_boxplot(
    dataset_root: str | Path,
    cache_root: str | Path,
    peds: Iterable[str] = ("UCSDped1", "UCSDped2"),
    save_path: str | Path | None = None,
    *,
    log_y: bool = True,
) -> "plt.Figure":
    """Box-plot of raw per-frame ‖S_t‖₁, split by dataset × normal/anomal.

    Visualises the asymmetry hypothesis: if Ped2 has a larger normal/anomal
    separation in ‖S‖₁ than Ped1, the boxes confirm that the RPCA-derived
    sparse component already carries most of the discriminative signal on
    Ped2 but not on Ped1.
    """
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)

    data = []
    labels = []
    for ped in peds:
        split_dir = dataset_root / ped / "Test"
        clip_dirs = list_clips(split_dir, kind="test")
        names = [c.name for c in clip_dirs]
        counts = clip_frame_counts(split_dir, kind="test")
        gt = parse_gt_m(split_dir / f"{ped}.m", n_frames_per_clip=counts)
        normal_vals = []
        anom_vals = []
        for n, lab in zip(names, gt):
            s = _per_clip_s_l1(cache_root, ped, "Test", n)
            mask = lab.astype(bool)
            normal_vals.append(s[~mask])
            anom_vals.append(s[mask])
        data.append(np.concatenate(normal_vals))
        data.append(np.concatenate(anom_vals))
        labels.append(f"{ped}\nnormal")
        labels.append(f"{ped}\nanomal")

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    widths=0.55)
    palette = ["#5b9bd5", "#ee5454"] * len(list(peds))
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_ylabel("‖S_t‖₁")
    ax.set_title("Per-frame ‖S‖₁ distribution: normal vs anomalous")
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Singular value spectrum
# ---------------------------------------------------------------------------

def plot_singular_spectrum(
    cache_root: str | Path,
    ped: str,
    split: str,
    clip_name: str,
    save_path: str | Path | None = None,
    *,
    k: int = 20,
) -> "plt.Figure":
    """Top-k singular values of the unfolded clip matrix X.

    Marks the rank that RPCA's L converged to (read from the cached metadata)
    so the reader sees that L is genuinely low-rank.
    """
    p = Path(cache_root) / ped / split / f"{clip_name}.npz"
    with np.load(p) as z:
        X_uint = z["X"]                         # (T, H, W) uint8
        rank = int(z["rank"]) if "rank" in z.files else -1
    X = (X_uint.astype(np.float32) / 255.0).reshape(X_uint.shape[0], -1).T  # (D, T)
    sigma = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    k_eff = min(k, len(sigma))

    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(1, k_eff + 1)
    ax.semilogy(idx, sigma[:k_eff], "o-", color="tab:blue", label="σ_i(X)")
    if rank > 0 and rank <= k_eff:
        ax.axvline(rank + 0.5, color="tab:red", ls="--", lw=1.2,
                   label=f"rank(L) = {rank}")
    ax.set_xlabel("index i")
    ax.set_ylabel("singular value (log)")
    ax.set_title(f"Singular spectrum of X — {ped}/{split}/{clip_name}")
    ax.grid(which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Per-clip AUC bar
# ---------------------------------------------------------------------------

def plot_per_clip_auc_bar(
    runs: list[dict],
    clip_names: Sequence[str],
    title: str,
    save_path: str | Path | None = None,
    *,
    label_key: str = "variant",
) -> "plt.Figure":
    """Grouped bar chart: per-clip AUC × variant.

    Reveals whether the gain from any variant is consistent across clips or
    driven by a few. Annotated with the median per-clip AUC per variant.
    """
    n_clips = len(clip_names)
    n_runs = len(runs)
    width = 0.8 / max(n_runs, 1)
    x = np.arange(n_clips)

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * n_clips), 4))
    colors = plt.cm.tab10.colors
    for i, r in enumerate(runs):
        aucs = np.asarray(r["per_clip_aucs"], dtype=np.float32)
        if len(aucs) != n_clips:
            # Defensive truncate/pad to expected length.
            aucs = np.resize(aucs, n_clips)
        offset = (i - (n_runs - 1) / 2) * width
        bars = ax.bar(x + offset, aucs, width=width * 0.95,
                      label=r[label_key], color=colors[i % len(colors)],
                      edgecolor="black", linewidth=0.4)
        med = float(np.nanmedian(aucs))
        ax.axhline(med, color=colors[i % len(colors)], lw=0.8, alpha=0.5,
                   ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels(clip_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncols=min(n_runs, 3))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 9. RPCA convergence
# ---------------------------------------------------------------------------

def plot_rpca_convergence(
    history: Sequence[float],
    title: str = "RPCA convergence",
    save_path: str | Path | None = None,
    *,
    tol: float | None = None,
) -> "plt.Figure":
    """Log-y plot of the per-iteration primal residual ‖X − L − S‖_F / ‖X‖_F."""
    h = np.asarray(history, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(h) + 1), h, "o-", color="tab:blue", lw=1.2)
    if tol is not None:
        ax.axhline(tol, color="tab:red", ls="--", lw=1.0,
                   label=f"tol = {tol:.0e}")
        ax.legend(loc="upper right")
    ax.set_xlabel("iteration")
    ax.set_ylabel("‖X − L − S‖_F / ‖X‖_F")
    ax.set_title(title)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, save_path)
    return fig
