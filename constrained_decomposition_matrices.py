"""
Matrix and basis generators used for demos/experiments.

This module contains:
  - SPD matrix generators (Hilbert, AR(1) Toeplitz, Brownian, Gaussian kernel, fractional BM, ...)
  - Convenient basis constructors for structured subspaces (banded, selected off-diagonals, block patterns, ...)
  - Small helpers for block/group demo construction

It should not contain solvers or plotting.
"""


import numpy as np

from constrained_decomposition_core import SymBasis



def spd_hilbert(n):
    """Hilbert matrix, dense SPD."""
    i = np.arange(1, n+1)
    j = np.arange(1, n+1)
    return 1.0 / (i[:, None] + j[None, :] - 1)

def spd_toeplitz_ar1(n: int, rho: float = 0.8, sigma2: float = 1.0) -> np.ndarray:
    """
    SPD Toeplitz covariance/correlation matrix:
        A[i,j] = sigma2 * rho^{|i-j|}
    This is the covariance of a stationary AR(1) process (up to scaling).
    Guaranteed SPD for any n when abs(rho) < 1 and sigma2 > 0.
    """
    if not (isinstance(n, int) and n >= 1):
        raise ValueError("n must be a positive integer")
    if not (abs(rho) < 1.0):
        raise ValueError("Need abs(rho) < 1 for SPD Toeplitz AR(1)")
    if not (sigma2 > 0):
        raise ValueError("Need sigma2 > 0")

    idx = np.arange(n)
    # Toeplitz via |i-j|
    A = sigma2 * (rho ** np.abs(idx[:, None] - idx[None, :]))
    # Symmetry is exact, but keep it clean numerically:
    return  0.5 * (A + A.T)

def spd_brownian(n):
    """Brownian-motion covariance matrix: K_ij = min(i,j)."""
    i = np.arange(1, n+1)
    j = np.arange(1, n+1)
    return np.minimum(i[:, None], j[None, :]).astype(float)

def spd_gaussian_kernel(n, gamma=0.1):
    """Gaussian kernel matrix: e^{-gamma (i-j)^2}."""
    i = np.arange(n)
    j = np.arange(n)
    return np.exp(-gamma * (i[:, None] - j[None, :])**2)

def spd_fractional_BM(n, H=0.5, T=1.0, diff_flag=False):
    """
    Fractional Brownian motion covariance matrix on an equispaced grid.

    A[i,j] = 0.5 * (T/n)^(2H) * ( i^(2H) + j^(2H) - |i-j|^(2H) )
    with i,j = 1,...,n (1-based indices in the formula).

    For diff:
    A[i,j] = 0.5 * (T/n)^(2H) * ( (i-j-1)^(2H) + (i-j+1)^(2H) - 2*|i-j|^(2H) )

    Parameters
    ----------
    n : int
        Matrix size.
    H : float, default 0.5
        Hurst parameter in (0,1).
    T : float, default 1.0
        Final time horizon.
    diff_flag: Show covariance of differences
    """
    i = np.arange(1, n + 1, dtype=float)
    j = np.arange(1, n + 1, dtype=float)
    I = i[:, None]
    J = j[None, :]

    factor = 0.5 * (T / n) ** (2.0 * H)

    if not diff_flag:
        A = factor * (I ** (2.0 * H) + J ** (2.0 * H) - np.abs(I - J) ** (2.0 * H))
    else:
        A = factor * (np.abs(I - J - 1) ** (2.0 * H) + np.abs(I - J + 1) ** (2.0 * H) - 2.0 * np.abs(I - J) ** (2.0 * H))
    return A





def make_random_spd(n: int, seed: int = 0, diag_boost: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + (diag_boost * n) * np.eye(n)
    return 0.5 * (A + A.T)

def make_banded_spd(n: int, b: int, seed: int = 0, diag_boost: float = 5.0) -> np.ndarray:
    """
    Dense representation of a symmetric banded SPD matrix (bandwidth b).
    (In code we still store dense; structure is in the pattern.)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=float)
    for k in range(b + 1):
        v = rng.standard_normal(n - k)
        if k == 0:
            A += np.diag(v)
        else:
            A += np.diag(v, k) + np.diag(v, -k)
    # boost diagonal to ensure SPD
    A += (diag_boost * (b + 1)) * np.eye(n)
    return 0.5 * (A + A.T)

def make_offdiag_pair_basis(n: int, pairs):
    mats = []
    for (i, j) in pairs:
        D = np.zeros((n, n), dtype=float)
        D[i, j] = 1.0
        D[j, i] = 1.0
        mats.append(D)
    return SymBasis(n=n, dense_mats=mats, name=f"offdiag_pairs_m={len(mats)}")

def make_banded_basis(n: int, b: int, include_diag: bool = True):
    """
    Basis for symmetric banded matrices with half-bandwidth b.
    include_diag=True => includes diagonal E_ii.
    """
    mats = []
    if include_diag:
        for i in range(n):
            D = np.zeros((n, n), dtype=float)
            D[i, i] = 1.0
            mats.append(D)
    for k in range(1, b + 1):
        for i in range(n - k):
            j = i + k
            D = np.zeros((n, n), dtype=float)
            D[i, j] = 1.0
            D[j, i] = 1.0
            mats.append(D)
    return SymBasis(n, dense_mats=mats, name=f"banded(b={b}, diag={include_diag})")

def block_reynolds_project(A: np.ndarray, blocks):
    """
    Project a matrix onto the block-permutation fixed space:
    entries become constant on each (block_i, block_j) rectangle.
    """
    n = A.shape[0]
    A = 0.5 * (A + A.T)
    out = np.zeros_like(A)
    for bi, I in enumerate(blocks):
        for bj, J in enumerate(blocks):
            sub = A[np.ix_(I, J)]
            out[np.ix_(I, J)] = np.mean(sub)
    out = 0.5 * (out + out.T)
    return out

def make_blocks(n: int, r: int):
    """
    Partition {0,...,n-1} into r contiguous blocks (sizes differ by at most 1).
    """
    r = int(r)
    if r < 1 or r > n:
        raise ValueError("blocks r must satisfy 1 <= r <= n")
    sizes = [n // r] * r
    for t in range(n % r):
        sizes[t] += 1
    blocks = []
    start = 0
    for sz in sizes:
        blocks.append(list(range(start, start + sz)))
        start += sz
    return blocks

def make_block_fixed_spd(n: int, r: int, seed: int = 0):
    """
    Create a random SPD A and Reynolds-project it to be fixed under block-permutations.
    """
    A0 = make_random_spd(n, seed=seed, diag_boost=2.0)
    blocks = make_blocks(n, r)
    A = block_reynolds_project(A0, blocks)
    # ensure strictly SPD by diagonal shift if needed
    lam_min = float(np.min(np.linalg.eigvalsh(A)))
    if lam_min <= 1e-8:
        A = A + (abs(lam_min) + 1e-2) * np.eye(n)
    return A, blocks

def make_block_fixed_basis_offdiag(n: int, blocks):
    """
    Build a *small* basis for S^G consisting of block-constant OFF-DIAGONAL patterns:
      - one matrix per block: within-block off-diagonal entries = 1 (diag=0)
      - one matrix per block-pair: between-block entries = 1 (both rectangles)
    This ensures S ∩ SPSD = {0} (zero diagonal).
    """
    mats = []

    # within-block off-diagonal
    for I in blocks:
        D = np.zeros((n, n), dtype=float)
        for a in I:
            for b in I:
                if a != b:
                    D[a, b] = 1.0
        mats.append(D)

    # between blocks
    for bi in range(len(blocks)):
        for bj in range(bi + 1, len(blocks)):
            I = blocks[bi]
            J = blocks[bj]
            D = np.zeros((n, n), dtype=float)
            for a in I:
                for b in J:
                    D[a, b] = 1.0
                    D[b, a] = 1.0
            mats.append(D)

    return SymBasis(n=n, dense_mats=mats, name=f"block_fixed_offdiag_r={len(blocks)}_m={len(mats)}")
