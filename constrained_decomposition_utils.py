import numpy as np
import matplotlib.pyplot as plt

import time
import math

def plot_decomposition_heatmaps(A, B, C, filename=None):
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
    Binv = np.linalg.inv(B)
    A_reconstructed = Binv + C
    err = np.linalg.norm(A - A_reconstructed, ord="fro")

    # Compute gradient diagnostic g_k = Bkk - Bk,k+1
    n = B.shape[0]
    g = np.array([B[k, k] - B[k, k+1] for k in range(n - 1)])
    sum_g = np.sum(g)
    max_g = np.max(np.abs(g))

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

def is_spd(M, tol=1e-12):
    """Check symmetric positive definite via Cholesky."""
    M = np.asarray(M, dtype=float)
    if not np.allclose(M, M.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False



def build_C_from_x(x):
    """
    Given x in R^{n-1}, build the tridiagonal C(x) satisfying:
      C_{kk} = x_k
      C_{k,k+1} = C_{k+1,k} = -x_k / 2
      C_{nn} = 0
    Here indices are 0-based in Python.
    """
    x = np.asarray(x, dtype=float)
    n_minus_1 = x.shape[0]
    n = n_minus_1 + 1
    C = np.zeros((n, n), dtype=float)

    for k in range(n_minus_1):
        C[k, k] = x[k]
        C[k, k + 1] = -0.5 * x[k]
        C[k + 1, k] = -0.5 * x[k]
    # C[n-1, n-1] already 0
    return C


def phi_and_grad_spd(A, x):
    """
    Phi(x) = -log det(A - C(x))  with A - C(x) required SPD.
    Returns: phi, grad (length n-1), B, C, M
    """
    C = build_C_from_x(x)
    M = A - C

    # Cholesky for SPD + logdet
    L = np.linalg.cholesky(M)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    phi = -logdet

    # B = M^{-1}
    # cheaper than np.linalg.inv(M) for repeated solves: but here we just do inv
    B = spd_inverse(M)

    n = A.shape[0]
    grad = np.empty(n - 1, dtype=float)
    for k in range(n - 1):
        grad[k] = B[k, k] - B[k, k + 1]

    return phi, grad, B, C, M


def phi_and_grad_general(A, x):
    """
    Phi(x) = -log |det(A - C(x))|, allowing A - C(x) to be indefinite
    as long as it is invertible.
    Returns: phi, grad (length n-1), B, C, M
    """
    C = build_C_from_x(x)
    M = A - C

    # det and log |det|
    sign, logabsdet = np.linalg.slogdet(M)
    if sign == 0:
        raise np.linalg.LinAlgError("A - C(x) is singular.")
    phi = -logabsdet  # use -log |det|

    B = spd_inverse(M)

    n = A.shape[0]
    grad = np.empty(n - 1, dtype=float)
    for k in range(n - 1):
        grad[k] = B[k, k] - B[k, k + 1]

    return phi, grad, B, C, M



def hessian_phi_spd_from_B(B):
    """
    Structured Hessian of Phi(x) = -log det(A - C(x)) in the SPD case,
    using only entries of B = (A - C(x))^{-1}.

    H_{kl} = Tr(B D^{(l)} B D^{(k)}),
    where D^{(k)} has support on indices {k,k+1}.

    Input
    -----
    B : (n,n) ndarray, symmetric positive definite

    Returns
    -------
    H : (n-1,n-1) ndarray
        Hessian in x-coordinates.
    """
    B = np.asarray(B, dtype=float)
    n = B.shape[0]
    m = n - 1
    H = np.zeros((m, m), dtype=float)

    # Diagonal blocks
    for k in range(m):
        Bkk = B[k, k]
        Bk_k1 = B[k, k + 1]
        Bk1_k1 = B[k + 1, k + 1]
        H[k, k] = (
            Bkk**2
            - 2.0 * Bkk * Bk_k1
            + 0.5 * Bkk * Bk1_k1
            + 0.5 * Bk_k1**2
        )

    # Off-diagonal blocks
    for k in range(m):
        # Neighbor l = k+1 (only if k+2 exists)
        l = k + 1
        if l < m and (k + 2) < n:
            Bk_k1 = B[k, k + 1]
            Bk_k2 = B[k, k + 2]
            Bk1_k1 = B[k + 1, k + 1]
            Bk1_k2 = B[k + 1, k + 2]
            val = (
                Bk_k1**2
                - Bk_k1 * Bk_k2
                - Bk_k1 * Bk1_k1
                + 0.5 * Bk_k1 * Bk1_k2
                + 0.5 * Bk_k2 * Bk1_k1
            )
            H[k, l] = H[l, k] = val

        # Non-neighbor l >= k+2
        for l in range(k + 2, m):
            # Here l+1 <= n-1 automatically
            Bk_l = B[k, l]
            Bk_l1 = B[k, l + 1]
            Bk1_l = B[k + 1, l]
            Bk1_l1 = B[k + 1, l + 1]
            val = (
                Bk_l**2
                - Bk_l * Bk_l1
                - Bk_l * Bk1_l
                + 0.5 * Bk_l * Bk1_l1
                + 0.5 * Bk_l1 * Bk1_l
            )
            H[k, l] = H[l, k] = val

    return H


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
    A = 0.5 * (A + A.T)
    return A

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



