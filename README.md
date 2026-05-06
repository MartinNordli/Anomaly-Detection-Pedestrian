# RPCA-Enhanced Anomaly Detection on UCSD

End-to-end pipeline that decomposes UCSD pedestrian video clips into a
low-rank background `L` and a sparse foreground `S` via Robust PCA, then
trains a small convolutional autoencoder for frame-level anomaly detection.
Compares four input representations on the same architecture: `raw`, `L`,
`S`, `[L, S]`.

See `Project_Description.md` for motivation and research questions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset already lives under `UCSD_Anomaly_Dataset.v1p2/` (Ped1 + Ped2
TIFF clips and `.m` ground-truth files).

## Run

Open the notebook:

```bash
jupyter lab notebooks/main.ipynb
```

The notebook drives the full pipeline cell by cell:
1. Sanity-check the data loader and parse frame-level GT.
2. Decompose one clip and visualize `X / L / |S|`.
3. Pre-compute and cache RPCA decompositions for every Ped2 clip
   (`cache/UCSDped2/{Train,Test}/<clip>.npz`).
4. Train one autoencoder per representation on Ped2 (≈10 min each on M4 MPS).
5. Score test frames by reconstruction error, report frame-level AUC/EER/ROC,
   and save a comparison table to `results/auc_table.csv`.
6. Repeat steps 3–5 for Ped1 (RQ3).

The notebook is reproducible from `scripts/build_notebook.py`:

```bash
python3 scripts/build_notebook.py
```

## Project layout

```
src/                  per-module source-of-truth (data, rpca, cache, dataset,
                      model, train, evaluate, viz)
notebooks/main.ipynb  end-to-end driver
scripts/              build_notebook.py
cache/                RPCA outputs (gitignored)
checkpoints/          trained model weights (gitignored)
results/              AUC table, ROC plots, qualitative figures (gitignored)
```

## Hardware notes

Tuned for an M4 MacBook Pro (10-core, 16 GB):
- Training runs on MPS via PyTorch (`device='mps'`).
- RPCA runs on CPU NumPy with randomized SVD (Apple Accelerate BLAS is fast
  for these matrix sizes).
- RPCA caches store `L`, `S` as float16 to halve disk usage.
- Per-clip independent decomposition keeps memory ≪ 1 GB at any time.

Approximate end-to-end runtime on M4: ~3 minutes for all RPCA caches, ~10
minutes per training run, four runs per ped → roughly 1.5 hours for the full
Ped1+Ped2 sweep.
