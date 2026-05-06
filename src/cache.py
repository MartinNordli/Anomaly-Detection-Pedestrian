"""Pre-decompose every UCSD clip with RPCA and persist L, S to disk.

Each clip becomes one .npz file under cache/<ped>/<split>/<ClipName>.npz with
keys 'L' and 'S' stored as float16 (halves disk vs float32 with negligible
quality loss; we cast back to float32 at load time). The original frames are
also cached as 'X' (uint8) so the dataset class can fetch the raw variant
without re-reading the TIFFs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .data import list_clips, load_clip_frames
from .rpca import rpca_clip


def _cache_path(cache_root: Path, ped: str, split: str, clip_name: str) -> Path:
    return cache_root / ped / split / f"{clip_name}.npz"


def precompute_split(
    dataset_root: str | Path,
    cache_root: str | Path,
    ped: str,
    split: str,
    *,
    force: bool = False,
    rpca_kwargs: dict | None = None,
) -> list[dict]:
    """Run RPCA over every clip in <ped>/<split>; cache L, S, X to .npz.

    Args:
        dataset_root: path containing UCSDped1/ and UCSDped2/.
        cache_root: where to write per-clip .npz files.
        ped: 'UCSDped1' or 'UCSDped2'.
        split: 'Train' or 'Test'.
        force: redo RPCA even if a cache file already exists.
        rpca_kwargs: passed through to rpca_clip (e.g. max_iter, tol).

    Returns:
        Per-clip diagnostic dicts (clip name, iters, residual, rank, time).
    """
    rpca_kwargs = rpca_kwargs or {}
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    split_dir = dataset_root / ped / split
    kind = "train" if split.lower().startswith("train") else "test"
    clips = list_clips(split_dir, kind=kind)

    out_dir = cache_root / ped / split
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: list[dict] = []
    for clip_dir in tqdm(clips, desc=f"RPCA {ped}/{split}"):
        out_path = _cache_path(cache_root, ped, split, clip_dir.name)
        if out_path.exists() and not force:
            with np.load(out_path) as z:
                diagnostics.append({
                    "clip": clip_dir.name,
                    "iters": int(z.get("iters", -1)),
                    "residual": float(z.get("residual", -1.0)),
                    "rank": int(z.get("rank", -1)),
                    "time_s": float(z.get("time_s", -1.0)),
                    "T": int(z["L"].shape[0]),
                    "cached": True,
                })
            continue

        frames = load_clip_frames(clip_dir)  # (T, H, W) float32 [0,1]
        L, S, info = rpca_clip(frames, **rpca_kwargs)
        np.savez_compressed(
            out_path,
            L=L.astype(np.float16),
            S=S.astype(np.float16),
            X=(frames * 255.0).astype(np.uint8),  # 8-bit raw — exact, smaller
            iters=np.int32(info.iters),
            residual=np.float32(info.final_residual),
            rank=np.int32(info.rank),
            time_s=np.float32(info.wall_time_s),
        )
        diagnostics.append({
            "clip": clip_dir.name,
            "iters": info.iters,
            "residual": info.final_residual,
            "rank": info.rank,
            "time_s": info.wall_time_s,
            "T": frames.shape[0],
            "cached": False,
        })
    return diagnostics


def precompute_all(
    dataset_root: str | Path,
    cache_root: str | Path,
    peds: Iterable[str] = ("UCSDped2", "UCSDped1"),
    splits: Iterable[str] = ("Train", "Test"),
    force: bool = False,
    rpca_kwargs: dict | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Run precompute_split over the cartesian product of peds × splits."""
    out: dict[tuple[str, str], list[dict]] = {}
    for ped in peds:
        for split in splits:
            out[(ped, split)] = precompute_split(
                dataset_root, cache_root, ped, split,
                force=force, rpca_kwargs=rpca_kwargs,
            )
    return out


def load_cached(cache_root: str | Path, ped: str, split: str, clip_name: str
                ) -> dict[str, np.ndarray]:
    """Memory-map a clip's cache. Returns dict with X (uint8), L, S (float16)."""
    p = _cache_path(Path(cache_root), ped, split, clip_name)
    z = np.load(p, mmap_mode="r")
    return {"X": z["X"], "L": z["L"], "S": z["S"]}
