"""Generate notebooks/main.ipynb programmatically.

Run from project root:
    python3 scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s)


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s)


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# RPCA-Enhanced Anomaly Detection on UCSD\n\n"
        "Two phases:\n\n"
        "**Phase 1 (baseline)** — per-frame ConvAE reconstruction. Establishes "
        "the RPCA cache and a weak baseline (~0.67 AUC on Ped2).\n\n"
        "**Phase 2 (predictor)** — UNet future-frame prediction with RPCA "
        "integrated four ways. Targets SOTA-territory AUC (≥ 0.85). RPCA's "
        "value is now testable because the task — predicting motion — is one "
        "where background subtraction is unambiguously useful.\n\n"
        "Plus an SVD weight-compression ablation (RQ4) on the trained predictor."
    ))

    cells.append(code(
        "import sys, os, time\n"
        "from pathlib import Path\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n"
        "DATA_ROOT = PROJECT_ROOT / 'UCSD_Anomaly_Dataset.v1p2'\n"
        "CACHE_ROOT = PROJECT_ROOT / 'cache'\n"
        "CKPT_ROOT = PROJECT_ROOT / 'checkpoints'\n"
        "RESULTS = PROJECT_ROOT / 'results'\n"
        "for p in (CACHE_ROOT, CKPT_ROOT, RESULTS, RESULTS / 'viz'):\n"
        "    p.mkdir(parents=True, exist_ok=True)\n\n"
        "import numpy as np, torch, matplotlib.pyplot as plt\n"
        "from src.data import parse_gt_m, list_clips, load_clip_frames, clip_frame_counts\n"
        "from src.rpca import rpca_clip\n"
        "from src.cache import precompute_split, load_cached\n"
        "from src.viz import visualize_decomposition\n\n"
        "DEVICE = torch.device('mps' if torch.backends.mps.is_available()\n"
        "                       else ('cuda' if torch.cuda.is_available() else 'cpu'))\n"
        "print('device:', DEVICE, '  torch:', torch.__version__)"
    ))

    cells.append(md("## 1. Data sanity checks"))

    cells.append(code(
        "ped = 'UCSDped2'\n"
        "counts = clip_frame_counts(DATA_ROOT / ped / 'Test', kind='test')\n"
        "labels = parse_gt_m(DATA_ROOT / ped / 'Test' / f'{ped}.m', n_frames_per_clip=counts)\n"
        "for i, (c, lab) in enumerate(zip(counts, labels)):\n"
        "    print(f'  clip {i+1:02d}  T={c:3d}  anom={lab.mean():.2f}  first_anom_idx={int(np.argmax(lab))}')"
    ))

    cells.append(code(
        "clip_dir = list_clips(DATA_ROOT / 'UCSDped2' / 'Train', kind='train')[0]\n"
        "frames = load_clip_frames(clip_dir)\n"
        "print(clip_dir.name, frames.shape, frames.dtype, 'min/max:', frames.min(), frames.max())\n"
        "fig, ax = plt.subplots(1, 3, figsize=(9, 3))\n"
        "for k, fi in enumerate([0, frames.shape[0]//2, frames.shape[0]-1]):\n"
        "    ax[k].imshow(frames[fi], cmap='gray'); ax[k].set_title(f'frame {fi}'); ax[k].axis('off')\n"
        "plt.show()"
    ))

    cells.append(md("## 2. RPCA decomposition (single-clip sanity check)"))

    cells.append(code(
        "L, S, info = rpca_clip(frames, max_iter=80, tol=1e-6, verbose=False)\n"
        "print(f'iters={info.iters}  residual={info.final_residual:.2e}  rank={info.rank}  '\n"
        "      f'time={info.wall_time_s:.2f}s')\n"
        "fi = frames.shape[0] // 2\n"
        "fig, ax = plt.subplots(1, 3, figsize=(11, 4))\n"
        "ax[0].imshow(frames[fi], cmap='gray', vmin=0, vmax=1); ax[0].set_title('X (raw)')\n"
        "ax[1].imshow(L[fi], cmap='gray'); ax[1].set_title('L (low rank)')\n"
        "im = ax[2].imshow(np.abs(S[fi]), cmap='hot'); ax[2].set_title('|S| (sparse)')\n"
        "plt.colorbar(im, ax=ax[2], fraction=0.04)\n"
        "for a in ax: a.axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(md("## 3. Bulk RPCA preprocessing (cached)"))

    cells.append(code(
        "for ped in ('UCSDped2', 'UCSDped1'):\n"
        "    for split in ('Train', 'Test'):\n"
        "        diags = precompute_split(DATA_ROOT, CACHE_ROOT, ped, split,\n"
        "                                 rpca_kwargs={'max_iter': 80, 'tol': 1e-6})\n"
        "        avg_iter = np.mean([d['iters'] for d in diags])\n"
        "        max_res = max(d['residual'] for d in diags)\n"
        "        print(f'  {ped}/{split}: {len(diags):>2} clips, avg_iter={avg_iter:.1f}, '\n"
        "              f'max_residual={max_res:.2e}')"
    ))

    cells.append(md(
        "## 4. Phase 2 — UNet future-frame predictor across four RPCA variants\n\n"
        "Same architecture and hyperparameters across the four variants — only "
        "the input/target tensors change. This isolates the contribution of RPCA."
    ))

    cells.append(code(
        "from src.dataset import VARIANTS\n"
        "from src.train import TrainConfig, train_predictor, load_checkpoint\n"
        "from src.evaluate import evaluate_run, append_results_csv\n"
        "from src.viz import plot_roc_overlay\n\n"
        "RESULTS_CSV = RESULTS / 'auc_table_v2.csv'\n"
        "PED = 'UCSDped2'\n"
        "cfg_p2 = TrainConfig(epochs=100, batch_size=16, lr=2e-4, num_workers=4,\n"
        "                     val_frac=0.15, early_stop_patience=10, augment=True,\n"
        "                     lambda_grad=1.0, base_channels=32, window=4, warmup_epochs=3)\n"
        "ped2_runs = []\n"
        "for v in VARIANTS:\n"
        "    ckpt = CKPT_ROOT / f'v2_{PED}_{v}.pt'\n"
        "    print(f'\\n=== {PED} / {v} ===')\n"
        "    train_predictor(DATA_ROOT, CACHE_ROOT, PED, v, ckpt,\n"
        "                    cfg=cfg_p2, device=DEVICE)\n"
        "    model = load_checkpoint(ckpt, device=DEVICE)\n"
        "    res = evaluate_run(model, DATA_ROOT, CACHE_ROOT, PED, v,\n"
        "                       device=DEVICE, window=cfg_p2.window, batch_size=16)\n"
        "    print(f\"  AUC_global={res['auc_global']:.4f}  AUC_per_clip={res['auc_per_clip']:.4f}  EER={res['eer']:.4f}\")\n"
        "    append_results_csv(RESULTS_CSV, res)\n"
        "    ped2_runs.append(res)"
    ))

    cells.append(code(
        "# ROC overlay across the four variants on Ped2.\n"
        "fig, ax = plt.subplots(figsize=(5, 5))\n"
        "for r in ped2_runs:\n"
        "    ax.plot(r['roc_fpr'], r['roc_tpr'],\n"
        "            label=f\"{r['variant']} (AUC={r['auc_global']:.3f})\")\n"
        "ax.plot([0,1], [0,1], 'k--', alpha=0.4); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')\n"
        "ax.set_title('UCSDped2 — frame-level ROC (predictor)')\n"
        "ax.legend(loc='lower right'); ax.grid(alpha=0.3)\n"
        "fig.tight_layout(); fig.savefig(RESULTS / 'viz' / 'roc_v2_ped2.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md("## 5. Same protocol on Ped1"))

    cells.append(code(
        "PED = 'UCSDped1'\n"
        "cfg_p1 = TrainConfig(epochs=100, batch_size=32, lr=2e-4, num_workers=4,\n"
        "                     val_frac=0.10, early_stop_patience=10, augment=True,\n"
        "                     lambda_grad=1.0, base_channels=32, window=4, warmup_epochs=3)\n"
        "ped1_runs = []\n"
        "for v in VARIANTS:\n"
        "    ckpt = CKPT_ROOT / f'v2_{PED}_{v}.pt'\n"
        "    print(f'\\n=== {PED} / {v} ===')\n"
        "    train_predictor(DATA_ROOT, CACHE_ROOT, PED, v, ckpt,\n"
        "                    cfg=cfg_p1, device=DEVICE)\n"
        "    model = load_checkpoint(ckpt, device=DEVICE)\n"
        "    res = evaluate_run(model, DATA_ROOT, CACHE_ROOT, PED, v,\n"
        "                       device=DEVICE, window=cfg_p1.window, batch_size=32)\n"
        "    print(f\"  AUC_global={res['auc_global']:.4f}  AUC_per_clip={res['auc_per_clip']:.4f}  EER={res['eer']:.4f}\")\n"
        "    append_results_csv(RESULTS_CSV, res)\n"
        "    ped1_runs.append(res)\n\n"
        "fig, ax = plt.subplots(figsize=(5, 5))\n"
        "for r in ped1_runs:\n"
        "    ax.plot(r['roc_fpr'], r['roc_tpr'],\n"
        "            label=f\"{r['variant']} (AUC={r['auc_global']:.3f})\")\n"
        "ax.plot([0,1], [0,1], 'k--', alpha=0.4); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')\n"
        "ax.set_title('UCSDped1 — frame-level ROC (predictor)')\n"
        "ax.legend(loc='lower right'); ax.grid(alpha=0.3)\n"
        "fig.savefig(RESULTS / 'viz' / 'roc_v2_ped1.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 6. SVD weight-compression ablation (RQ4)\n\n"
        "Truncate every Conv2d weight matrix in the trained UNet at multiple "
        "ranks via SVD. The resulting AUC-vs-compression curve quantifies how "
        "much of the model's discriminative capacity actually lives in the "
        "top singular directions of its weights — a direct demonstration of "
        "SVD as model compression, complementing SVD's role inside RPCA's "
        "singular-value-thresholding step."
    ))

    cells.append(code(
        "from src.svd_compression import evaluate_compression_curve, append_compression_csv\n"
        "import pandas as pd\n\n"
        "# Pick the best Ped2 variant from the AUC table.\n"
        "best = max(ped2_runs, key=lambda r: r['auc_global'])\n"
        "PED, V = 'UCSDped2', best['variant']\n"
        "print(f'compressing {PED}/{V} (AUC={best[\"auc_global\"]:.4f})')\n"
        "model = load_checkpoint(CKPT_ROOT / f'v2_{PED}_{V}.pt', device=DEVICE)\n"
        "rows = evaluate_compression_curve(\n"
        "    model, DATA_ROOT, CACHE_ROOT, PED, V, device=DEVICE,\n"
        "    rank_fracs=(1.0, 0.75, 0.5, 0.25, 0.1), window=4, batch_size=16,\n"
        ")\n"
        "append_compression_csv(RESULTS / 'svd_compression.csv', PED, V, rows)\n"
        "df_svd = pd.DataFrame(rows)\n"
        "print(df_svd[['rank_frac','compression_ratio','auc_global','auc_per_clip']].to_string(index=False))"
    ))

    cells.append(code(
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(df_svd['compression_ratio'], df_svd['auc_global'], 'o-', label='global AUC')\n"
        "ax.plot(df_svd['compression_ratio'], df_svd['auc_per_clip'], 's-', label='per-clip AUC')\n"
        "ax.set_xlabel('effective parameters / full parameters')\n"
        "ax.set_ylabel('frame-level AUC')\n"
        "ax.set_title(f'SVD compression curve — {PED}/{V}')\n"
        "ax.invert_xaxis()\n"
        "ax.grid(alpha=0.3); ax.legend(loc='lower left')\n"
        "fig.tight_layout(); fig.savefig(RESULTS / 'viz' / 'svd_compression.png', dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md("## 7. Final summary — Phase 1 vs Phase 2"))

    cells.append(code(
        "import pandas as pd\n"
        "df_v2 = pd.read_csv(RESULTS_CSV)\n"
        "df_v2 = df_v2.sort_values(['ped','variant']).reset_index(drop=True)\n"
        "print('=== Phase 2 (predictor) ===')\n"
        "print(df_v2[['ped','variant','auc_global','auc_per_clip','eer']].to_string(index=False))\n"
        "df_v2.to_markdown(RESULTS / 'auc_table_v2.md', index=False)\n\n"
        "phase1 = RESULTS / 'auc_table.csv'\n"
        "if phase1.exists():\n"
        "    df_v1 = pd.read_csv(phase1)\n"
        "    print('\\n=== Phase 1 (per-frame AE, for comparison) ===')\n"
        "    print(df_v1[['ped','representation','auc_mean']].to_string(index=False))\n"
        "df_v2"
    ))

    cells.append(md("## 8. Qualitative L/S decomposition figures"))

    cells.append(code(
        "_ = visualize_decomposition(CACHE_ROOT, 'UCSDped2', 'Train', 'Train001',\n"
        "                            save_path=RESULTS / 'viz' / 'ped2_train001_decomp.png')\n"
        "_ = visualize_decomposition(CACHE_ROOT, 'UCSDped2', 'Test', 'Test001',\n"
        "                            frame_indices=[10, 40, 70, 100, 130, 170],\n"
        "                            save_path=RESULTS / 'viz' / 'ped2_test001_decomp.png')\n"
        "plt.show()"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "main.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
