"""UCSD Pedestrian Anomaly Dataset I/O.

Handles:
- Parsing the MATLAB-style frame-level GT files (UCSDped{1,2}.m).
- Listing clip directories while filtering macOS resource forks (._*) and
  pixel-level GT directories (*_gt).
- Loading a clip's TIFF frames into a contiguous float32 array in [0, 1].
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# Matches: TestVideoFile{end+1}.gt_frame = [<spec>];
_GT_LINE = re.compile(
    r"TestVideoFile\{[^}]+\}\.gt_frame\s*=\s*\[([^\]]+)\]\s*;"
)


def parse_gt_m(path: str | Path, n_frames_per_clip: list[int] | None = None
               ) -> list[np.ndarray]:
    """Parse a UCSDpedX.m ground-truth file into per-clip binary frame labels.

    Each line of the form `TestVideoFile{end+1}.gt_frame = [a:b, c:d];` becomes
    one entry in the returned list. The boolean array is True at frame indices
    that are anomalous.

    Args:
        path: path to UCSDped1.m or UCSDped2.m.
        n_frames_per_clip: optional list of per-clip frame counts (1-based clip
            order matching the .m file). If None, infers length from the max
            frame index referenced for that clip — sufficient when every clip
            ends with an anomalous frame, but pass explicit lengths to be safe.

    Returns:
        List of bool ndarrays, one per clip, shape (T_i,).
    """
    path = Path(path)
    text = path.read_text()
    specs: list[str] = _GT_LINE.findall(text)

    labels_per_clip: list[np.ndarray] = []
    for i, spec in enumerate(specs):
        # spec is e.g. "60:152" or "5:90, 140:200"
        ranges = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if ":" in chunk:
                a, b = chunk.split(":")
                ranges.append((int(a), int(b)))
            else:
                v = int(chunk)
                ranges.append((v, v))

        if n_frames_per_clip is not None:
            T = n_frames_per_clip[i]
        else:
            T = max(b for _, b in ranges)

        lab = np.zeros(T, dtype=bool)
        for a, b in ranges:
            # .m file is 1-indexed inclusive; convert to 0-indexed.
            lab[a - 1 : b] = True
        labels_per_clip.append(lab)
    return labels_per_clip


def list_clips(split_dir: str | Path, kind: str = "train") -> list[Path]:
    """Return sorted list of clip directories under a split.

    Args:
        split_dir: e.g. .../UCSDped2/Train or .../UCSDped2/Test
        kind: 'train' or 'test'. For 'test' we exclude *_gt pixel-mask dirs.
    """
    split_dir = Path(split_dir)
    clips: list[Path] = []
    for child in sorted(split_dir.iterdir()):
        name = child.name
        if name.startswith("."):
            continue  # .DS_Store, ._* resource forks
        if not child.is_dir():
            continue
        if kind == "test" and name.endswith("_gt"):
            continue
        if not name.startswith(("Train", "Test")):
            continue
        clips.append(child)
    return clips


def _read_tiff(path: Path) -> np.ndarray | None:
    """Try tifffile, fall back to PIL. Return None on total failure."""
    try:
        return tifffile.imread(str(path))
    except Exception:
        pass
    try:
        return np.array(Image.open(path))
    except Exception:
        return None


def load_clip_frames(clip_dir: str | Path) -> np.ndarray:
    """Load all TIFF frames in a clip dir into (T, H, W) float32 in [0, 1].

    The shipped UCSD Ped1 has at least one corrupt TIFF
    (UCSDped1/Test/Test017/142.tif). Such frames are substituted with the
    nearest valid frame and a warning is emitted; substitutions are also
    recorded on `load_clip_frames._substitutions` for downstream logging.
    """
    clip_dir = Path(clip_dir)
    files = sorted(
        f for f in clip_dir.iterdir()
        if f.suffix.lower() == ".tif" and not f.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No .tif frames in {clip_dir}")

    raw: list[np.ndarray | None] = [_read_tiff(f) for f in files]
    bad = [i for i, im in enumerate(raw) if im is None]
    if bad:
        # Reference shape from the first good frame.
        ref = next(im for im in raw if im is not None)
        for i in bad:
            # Use nearest non-None neighbor (prefer previous, then next).
            j_prev = next((k for k in range(i - 1, -1, -1) if raw[k] is not None), None)
            j_next = next((k for k in range(i + 1, len(raw)) if raw[k] is not None), None)
            j = j_prev if j_prev is not None else j_next
            raw[i] = raw[j].copy() if j is not None else np.zeros_like(ref)
            warnings.warn(
                f"corrupt TIFF {files[i]} replaced with frame {j} (nearest valid)",
                RuntimeWarning,
                stacklevel=2,
            )
        load_clip_frames._substitutions.append(  # type: ignore[attr-defined]
            (str(clip_dir), [files[i].name for i in bad])
        )

    frames = np.stack(raw, axis=0)  # type: ignore[arg-type]
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32) / 255.0
    return np.ascontiguousarray(frames)


load_clip_frames._substitutions = []  # type: ignore[attr-defined]


def clip_frame_counts(split_dir: str | Path, kind: str = "test") -> list[int]:
    """Return frame count per clip in a split, in clip order."""
    return [
        sum(1 for f in c.iterdir()
            if f.suffix.lower() == ".tif" and not f.name.startswith("."))
        for c in list_clips(split_dir, kind=kind)
    ]
