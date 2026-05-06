"""PyTorch Dataset for future-frame prediction on UCSD.

Each item is `(input_stack, target, label_at_target_t)` where:
  - `input_stack` is `window` consecutive frames at times t-window..t-1,
    stacked along the channel dimension.
  - `target` is the frame at time t (i.e. the next-frame prediction target).
  - `label_at_target_t` indicates whether frame t is anomalous (test only;
    train returns 0).

Four variants control which UCSD representation is used for input vs target:

    name        input             target
    raw         x_{t-w..t-1}      x_t
    S           S_{t-w..t-1}      S_t
    LS          (L,S)_{t-w..t-1}  S_t
    raw_to_S    x_{t-w..t-1}      S_t

S is min-max-normalised per clip (one shared scale across input and target),
which is essential because S has heavy tails and the per-clip statistics
match the static-camera assumption that motivates RPCA in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_cached
from .data import clip_frame_counts, list_clips, parse_gt_m


@dataclass(frozen=True)
class VariantSpec:
    """Maps a variant name to the (input_kind, target_kind) tuple."""
    name: str
    input_kind: tuple[str, ...]   # subset of ('X', 'L', 'S')
    target_kind: str              # one of 'X', 'L', 'S'

    @property
    def in_channels_per_frame(self) -> int:
        return len(self.input_kind)


VARIANTS: dict[str, VariantSpec] = {
    "raw":      VariantSpec("raw",      ("X",),       "X"),
    "S":        VariantSpec("S",        ("S",),       "S"),
    "LS":       VariantSpec("LS",       ("L", "S"),   "S"),
    "raw_to_S": VariantSpec("raw_to_S", ("X",),       "S"),
}


def _rescale01(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    a_min = float(arr.min())
    a_max = float(arr.max())
    if a_max - a_min < 1e-6:
        return np.zeros_like(arr, dtype=np.float32), a_min, a_max
    return ((arr.astype(np.float32) - a_min) / (a_max - a_min)).astype(np.float32), a_min, a_max


class UCSDPredictionDataset(Dataset):
    """Future-frame prediction dataset.

    Args:
        dataset_root: path containing UCSDpedX/.
        cache_root:   path containing cached RPCA .npz files.
        ped:          'UCSDped1' or 'UCSDped2'.
        split:        'Train' or 'Test'.
        variant:      one of VARIANTS keys.
        window:       number of past frames in the stack (default 4).
        clip_filter:  optional list of clip names to restrict to.
        return_label: if True, __getitem__ returns (stack, target, label_t).
                      Always True at test time (model evaluation needs labels).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        cache_root: str | Path,
        ped: str,
        split: str,
        variant: str = "raw",
        window: int = 4,
        clip_filter: list[str] | None = None,
        return_label: bool | None = None,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be in {list(VARIANTS)}")
        self.dataset_root = Path(dataset_root)
        self.cache_root = Path(cache_root)
        self.ped = ped
        self.split = split
        self.spec = VARIANTS[variant]
        self.window = window

        kind = "train" if split.lower().startswith("train") else "test"
        clip_dirs = list_clips(self.dataset_root / ped / split, kind=kind)
        if clip_filter is not None:
            keep = set(clip_filter)
            clip_dirs = [c for c in clip_dirs if c.name in keep]
        self.clip_names = [c.name for c in clip_dirs]

        # Per-clip frame counts.
        self.counts = [
            sum(1 for f in c.iterdir()
                if f.suffix.lower() == ".tif" and not f.name.startswith("."))
            for c in clip_dirs
        ]

        # Per-clip labels (test only).
        if kind == "test":
            full_counts = clip_frame_counts(self.dataset_root / ped / split, kind="test")
            full_labels = parse_gt_m(
                self.dataset_root / ped / split / f"{ped}.m",
                n_frames_per_clip=full_counts,
            )
            full_names = [c.name for c in list_clips(self.dataset_root / ped / split, kind="test")]
            name_to_label = dict(zip(full_names, full_labels))
            self.labels_per_clip = [name_to_label[n] for n in self.clip_names]
        else:
            self.labels_per_clip = [np.zeros(c, dtype=bool) for c in self.counts]

        if return_label is None:
            return_label = (kind == "test")
        self.return_label = return_label

        # Valid target indices per clip: t ∈ [window, T-1].
        self._index: list[tuple[int, int]] = [
            (ci, t) for ci, T in enumerate(self.counts) for t in range(window, T)
        ]
        self._cache: list[dict | None] = [None] * len(self.clip_names)

    def __len__(self) -> int:
        return len(self._index)

    def _ensure_loaded(self, ci: int) -> dict:
        """Load and pre-rescale this clip's cached blob."""
        if self._cache[ci] is None:
            blob = load_cached(self.cache_root, self.ped, self.split, self.clip_names[ci])
            entry: dict = {}
            need = set(self.spec.input_kind) | {self.spec.target_kind}
            if "X" in need:
                entry["X"] = blob["X"][:].astype(np.float32) / 255.0
            if "L" in need:
                Larr = blob["L"][:].astype(np.float32)
                # L is roughly in image range; clip and scale together with safety.
                L01, _, _ = _rescale01(Larr)
                entry["L"] = L01
            if "S" in need:
                Sarr = blob["S"][:].astype(np.float32)
                S01, _, _ = _rescale01(Sarr)
                entry["S"] = S01
            self._cache[ci] = entry
        return self._cache[ci]  # type: ignore[return-value]

    def _frame(self, entry: dict, kind: str, fi: int) -> np.ndarray:
        return entry[kind][fi]

    def __getitem__(self, idx: int):
        ci, t = self._index[idx]
        entry = self._ensure_loaded(ci)

        # Stack: t-window..t-1, channel-major: [(kind1, f0), (kind2, f0), ..., (kind1, f_{w-1}), ...]
        stack_planes: list[np.ndarray] = []
        for offset in range(self.window):
            fi = t - self.window + offset
            for kind in self.spec.input_kind:
                stack_planes.append(self._frame(entry, kind, fi))
        stack = np.stack(stack_planes, axis=0)          # (window * Cf, H, W)
        target = self._frame(entry, self.spec.target_kind, t)[None]  # (1, H, W)

        stack_t = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32))
        target_t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
        if self.return_label:
            y = bool(self.labels_per_clip[ci][t])
            return stack_t, target_t, torch.tensor(y, dtype=torch.float32)
        return stack_t, target_t

    @property
    def in_channels(self) -> int:
        return self.window * self.spec.in_channels_per_frame

    @property
    def out_channels(self) -> int:
        return 1


def split_train_clips(clip_names: list[str], val_frac: float = 0.15, seed: int = 0
                      ) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(clip_names) * val_frac)))
    perm = rng.permutation(len(clip_names))
    val_idx = set(perm[:n_val].tolist())
    train = [c for i, c in enumerate(clip_names) if i not in val_idx]
    val = [c for i, c in enumerate(clip_names) if i in val_idx]
    return train, val
