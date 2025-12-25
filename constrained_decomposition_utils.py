import numpy as np
import matplotlib.pyplot as plt

import time
import math

def plot_decomposition_heatmaps(A, B, C, basis, filename=None, add_title=False):
    """
    Plot 2x2 heatmaps:
        top-left:  A
        top-right: C
        bottom-left:  B
        bottom-right: B^{-1}

    Title includes:
        - Reconstruction error
        - Sum of (Bkk - Bk,k+1)
        - Max |Bkk - Bk,k+1|
        - Phi = -log det(A - C)

    Everything printed with 3 significant digits.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    # Compute inverse and reconstruction error
    Binv = spd_inverse(B)
    A_reconstructed = Binv + C
    err = np.linalg.norm(A - A_reconstructed, ord="fro")

    # Compute gradient diagnostic g_k = Bkk - Bk,k+1
    n = B.shape[0]

    g = None
    if basis is not None and hasattr(basis, "trace_with"):
        try:
            g = basis.trace_with(B)
        except Exception:
            g = None

    if g is not None:
        sum_g = np.sum(g)
        max_g = np.max(np.abs(g))
    else:
        sum_g = np.nan
        max_g = np.nan

    # Compute phi = -log det(A - C)
    M = A - C
    try:
        L = np.linalg.cholesky(M)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        phi = -logdet
    except np.linalg.LinAlgError:
        phi = np.nan  # If not SPD, logdet undefined

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    matrices = [[A, C],
                [B, Binv]]
    titles = [["A", "C"],
              ["B", r"$B^{-1}$"]]

    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            im = ax.imshow(matrices[r][c], aspect="equal")
            ax.set_title(titles[r][c])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Title: small font and multiple diagnostics ---
    if add_title:
        fig.suptitle(
            rf"Decomposition heatmaps"
            rf"\n$\|A-(B^{{-1}}+C)\|_F = {err:.3g}$"
            rf",  $\sum g_k = {sum_g:.3g}$"
            rf",  $\max |g_k| = {max_g:.3g}$"
            rf",  $\phi = {phi:.3g}$",
            fontsize=9
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig, axes

def is_spd(M, sym_tol=1e-12, jitter=0.0):
    """
    Robust SPD check:
      - symmetrize first (removes tiny asymmetry)
      - optional diagonal jitter (helps near-boundary SPD)
      - then Cholesky
    """
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)

    # Optional: if you still want a symmetry diagnostic, do it after symmetrizing
    # but don't use it as a hard failure condition.
    if sym_tol is not None:
        # relative-ish symmetry check
        if np.linalg.norm(M - M.T, ord="fro") > sym_tol * np.linalg.norm(M, ord="fro"):
            return False  # this is now basically never triggered after sym()

    n = M.shape[0]
    if jitter == "auto":
        jitter = 1e-12 * (np.trace(M) / n)  # scale-aware tiny diagonal bump

    try:
        np.linalg.cholesky(M + (jitter * np.eye(n) if jitter else 0.0))
        return True
    except np.linalg.LinAlgError:
        return False




######################################################
#### Matrix generators ###############################
######################################################



def spd_hilbert(n):
    """Hilbert matrix, dense SPD."""
    i = np.arange(1, n+1)
    j = np.arange(1, n+1)
    return 1.0 / (i[:, None] + j[None, :] - 1)



def spd_ar1(n, rho=0.9, sigma2=1.0):
    return spd_toeplitz_ar1(n, rho=rho, sigma2=sigma2)


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
    return  0.5 * (A + A.T)  # A

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


def spd_inverse(A):
    """
    Numerically stable inverse for SPD matrices using Cholesky solves.
    Returns a symmetrized inverse.
    """
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    L = np.linalg.cholesky(A)
    I = np.eye(A.shape[0])
    Y = np.linalg.solve(L, I)
    A_inv = np.linalg.solve(L.T, Y)
    return 0.5 * (A_inv + A_inv.T)



# -------- example usage --------



