"""Robust PCA via Inexact ALM (Lin, Chen, Ma 2010), accelerated with
randomized SVD (Halko, Martinsson, Tropp 2011).

Solves   min_{L,S}  ||L||_* + lambda ||S||_1   s.t.  X = L + S.

The IALM iteration alternates a singular-value-thresholding step on
(X - S + Y/mu) for L and a soft-thresholding step on (X - L + Y/mu) for S,
followed by a dual update of Y and a geometric increase of mu.

For tall-thin video matrices (D = H*W >> T) the dominant cost is the SVD
inside the SVT step. Replacing it with a truncated randomized SVD whose rank
adapts upward whenever singular values survive thresholding gives an order of
magnitude speedup with no loss of decomposition quality at the rank we need.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd


@dataclass
class RpcaInfo:
    iters: int
    final_residual: float
    rank: int
    sparsity: float
    wall_time_s: float
    history: list[float] | None = None


def _shrink(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def rpca_ialm(
    X: np.ndarray,
    lam: float | None = None,
    mu: float | None = None,
    rho: float = 1.5,
    tol: float = 1e-7,
    max_iter: int = 200,
    rsvd_oversamples: int = 10,
    rsvd_n_iter: int = 2,
    verbose: bool = False,
    record_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, RpcaInfo]:
    """Decompose X into low-rank L and sparse S via Inexact ALM with rSVD.

    Args:
        X: (D, T) matrix, float32.
        lam: sparsity weight; default 1/sqrt(max(D, T)) per Candes 2011.
        mu: initial penalty; default D*T / (4 ||X||_1).
        rho: geometric growth factor for mu.
        tol: relative-Frobenius stopping tolerance.
        max_iter: hard iteration cap.
        rsvd_oversamples, rsvd_n_iter: randomized SVD knobs.

    Returns:
        L, S (same shape as X, float32) and an RpcaInfo summary.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    D, T = X.shape
    norm_X = np.linalg.norm(X)
    if norm_X == 0:
        return np.zeros_like(X), np.zeros_like(X), RpcaInfo(0, 0.0, 0, 0.0, 0.0)

    if lam is None:
        lam = 1.0 / np.sqrt(max(D, T))
    if mu is None:
        mu = float(D * T / (4.0 * np.linalg.norm(X, ord=1)))
    mu_bar = mu * 1e7

    # Standard IALM dual init: Y = X / J(X) where J(X) = max(||X||_2, ||X||_inf / lam).
    # ||X||_2 ≈ top singular value; cheap via one rSVD.
    U0, s0, _ = randomized_svd(X, n_components=1, n_iter=2, random_state=0)
    norm_two = float(s0[0])
    norm_inf = float(np.max(np.abs(X)) / lam)
    Y = X / max(norm_two, norm_inf)

    L = np.zeros_like(X)
    S = np.zeros_like(X)

    sv = max(1, min(D, T) // 20)  # current truncation rank, grows adaptively
    sv_max = min(D, T)

    t0 = time.perf_counter()
    final_residual = float("inf")
    history: list[float] | None = [] if record_history else None
    it = 0
    for it in range(1, max_iter + 1):
        # --- L update: SVT on (X - S + Y/mu) at threshold 1/mu ---
        M = X - S + Y / mu
        n_components = min(int(sv) + rsvd_oversamples, sv_max)
        U, sigma, Vt = randomized_svd(
            M,
            n_components=n_components,
            n_iter=rsvd_n_iter,
            random_state=0,
        )
        thresh = 1.0 / mu
        keep = int((sigma > thresh).sum())
        # Adaptively grow sv if all rSVD components survived (we may be missing mass).
        if keep == n_components and n_components < sv_max:
            sv = min(sv_max, int(sv * 2))
        else:
            sv = max(1, keep + 1)
        if keep > 0:
            L = (U[:, :keep] * (sigma[:keep] - thresh)).astype(np.float32) @ Vt[:keep].astype(np.float32)
        else:
            L = np.zeros_like(X)

        # --- S update: soft-threshold (X - L + Y/mu) at lam/mu ---
        S = _shrink(X - L + Y / mu, lam / mu).astype(np.float32)

        # --- Dual + penalty updates ---
        residual = X - L - S
        Y = (Y + mu * residual).astype(np.float32)
        mu = min(mu * rho, mu_bar)

        rel = float(np.linalg.norm(residual) / norm_X)
        final_residual = rel
        if history is not None:
            history.append(rel)
        if verbose and (it % 10 == 0 or it == 1):
            print(f"iter {it:3d}  rel={rel:.2e}  rank={keep}  mu={mu:.3g}")
        if rel < tol:
            break

    sparsity = float((S != 0).mean())
    rank = int(np.linalg.matrix_rank(L)) if L.size else 0
    info = RpcaInfo(
        iters=it,
        final_residual=final_residual,
        rank=rank,
        sparsity=sparsity,
        wall_time_s=time.perf_counter() - t0,
        history=history,
    )
    return L, S, info


def rpca_clip(frames: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray, RpcaInfo]:
    """Run RPCA on a video clip given as (T, H, W) frames.

    Returns L, S each as (T, H, W), float32.
    """
    if frames.ndim != 3:
        raise ValueError(f"expected (T,H,W) got {frames.shape}")
    T, H, W = frames.shape
    X = frames.reshape(T, H * W).T.astype(np.float32, copy=False)  # (D, T)
    L, S, info = rpca_ialm(X, **kwargs)
    L = L.T.reshape(T, H, W)
    S = S.T.reshape(T, H, W)
    return L, S, info
