import numpy as np
import matplotlib.pyplot as plt
import time


def plot_decomposition_heatmaps(A, B, C, filename=None):
    """
    Plot 2x2 heatmaps:
        top-left:  A
        top-right: C
        bottom-left:  B
        bottom-right: B^{-1}

    The figure title contains the Frobenius norm error between A and B^{-1}+C.
    If filename is given, the figure is saved as a PNG.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    Binv = np.linalg.inv(B)
    A_reconstructed = Binv + C
    err = np.linalg.norm(A - A_reconstructed, ord="fro")

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    matrices = [[A, C],
                [B, Binv]]
    titles = [["A", "C"],
              ["B", "B$^{-1}$"]]

    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            im = ax.imshow(matrices[r][c], aspect="equal")
            ax.set_title(titles[r][c])
            # individual colorbar for each subplot
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(rf"Decomposition heatmaps, "
                 rf"$\|A - (B^{{-1}} + C)\|_F = {err:.3g}$")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig, axes


def is_spd(M, tol=1e-12):
    """
    Check if M is symmetric positive definite via Cholesky.
    """
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
    B = np.linalg.inv(M)

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

    B = np.linalg.inv(M)

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


def constrained_decomposition(
    A,
    allow_indefinite_B=False,
    tol=1e-8,
    max_iter=500,
    initial_step=1.0,
    backtracking_factor=0.5,
    armijo_alpha=1e-4,
    method="gradient-descent",
    verbose=False,
):
    """
    Compute B, C, x such that A ≈ B^{-1} + C, with C satisfying the
    tridiagonal structure and (approximately) B_{kk} = B_{k,k+1}.

    Parameters
    ----------
    A : (n,n) ndarray
        Symmetric positive-definite matrix.
    allow_indefinite_B : bool, default False
        If False: enforce A - C(x) SPD and use convex formulation
        (unique solution).
        If True: allow A - C(x) to be any invertible symmetric matrix and
        minimize -log|det(A - C(x))|; B may be indefinite.
        (Newton is only implemented for the SPD case, allow_indefinite_B=False.)
    tol : float
        Tolerance on max_k |B_{kk} - B_{k,k+1}| (and gradient norm scale).
    max_iter : int
        Maximum number of iterations.
    initial_step : float
        Initial step size for backtracking line search.
    backtracking_factor : float
        Factor in (0,1) by which step size is multiplied in backtracking.
    armijo_alpha : float
        Armijo parameter in (0,1) for sufficient decrease condition.
    method : {"gradient-descent", "quasi-newton", "newton"}
        Optimization method.
        - "gradient-descent": plain steepest descent in x.
        - "quasi-newton": BFGS in x.
        - "newton": full Newton using explicit Hessian (SPD case only).
    verbose : bool
        If True, print diagnostics during optimization.

    Returns
    -------
    B, C, x : ndarrays
        Decomposition A ≈ B^{-1} + C corresponding to final x.
    """

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), "A must be square"

    if not is_spd(A):
        raise ValueError("Input A is not symmetric positive definite.")

    if n == 1:
        C = np.zeros_like(A)
        B = np.linalg.inv(A)
        x = np.zeros(0)
        return B, C, x

    if method not in ("gradient-descent", "quasi-newton", "newton"):
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Use 'gradient-descent', 'quasi-newton', or 'newton'."
        )

    if method == "newton" and allow_indefinite_B:
        raise NotImplementedError(
            "Newton method is implemented only for the SPD formulation "
            "(allow_indefinite_B=False)."
        )

    # Initial x
    x = np.zeros(n - 1, dtype=float)

    # Initial phi, grad, B, C, M
    if allow_indefinite_B:
        phi, grad, B, C, M = phi_and_grad_general(A, x)
    else:
        phi, grad, B, C, M = phi_and_grad_spd(A, x)

    # For quasi-Newton (BFGS) we maintain an approximation H_BFGS ≈ (∇²Φ)^{-1}
    if method == "quasi-newton":
        H_BFGS = np.eye(n - 1)

    # For diagnostics over windows
    x_prev_window = x.copy()
    B_prev_window = B.copy()

    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad)

        # KKT residuals: should go to zero at optimum
        r = np.array([B[k, k] - B[k, k + 1] for k in range(n - 1)])
        max_r = np.max(np.abs(r))

        if verbose:
            eigvals_M = np.linalg.eigvalsh(M)
            lam_min = eigvals_M[0]
            print(
                f"iter {it:4d}, phi={phi: .6e}, ||grad||={grad_norm: .3e}, "
                f"max|Bkk-Bk,k+1|={max_r: .3e}, lambda_min(M)={lam_min: .3e}"
            )

        # Stopping based on your structural condition
        if max_r < tol:
            if verbose:
                print(
                    f"Converged: max|Bkk-Bk,k+1|={max_r:.3e} < tol={tol}"
                )
            break

        # Occasionally report change over a window
        if verbose and it > 0 and (it % 50 == 0):
            dx_window = np.linalg.norm(x - x_prev_window)
            dB_window = np.linalg.norm(B - B_prev_window, ord="fro")
            print(
                f"    over last 50 iters: ||Δx||={dx_window: .3e}, "
                f"||ΔB||_F={dB_window: .3e}"
            )
            x_prev_window = x.copy()
            B_prev_window = B.copy()

        # --- Compute search direction d ---

        if method == "gradient-descent":
            d = -grad

        elif method == "quasi-newton":
            d = -H_BFGS @ grad

        else:  # method == "newton"
            # Build structured Hessian from B
            H = hessian_phi_spd_from_B(B)
            # Solve H d = -grad via Cholesky; if fails, fall back to steepest descent
            try:
                Lh = np.linalg.cholesky(H)
                y = np.linalg.solve(Lh, -grad)
                d = np.linalg.solve(Lh.T, y)
            except np.linalg.LinAlgError:
                if verbose:
                    print("    Hessian not SPD; falling back to steepest descent.")
                d = -grad

        # Ensure descent direction; if not, fall back
        g = grad
        gTd = float(g @ d)
        if gTd >= 0:
            if verbose:
                print("    Non-descent direction encountered; using -grad instead.")
            d = -g
            gTd = float(g @ d)

        # Backtracking line search with general Armijo condition
        t = initial_step
        success = False

        for bt in range(50):
            x_try = x + t * d
            try:
                if allow_indefinite_B:
                    phi_try, grad_try, B_try, C_try, M_try = phi_and_grad_general(A, x_try)
                else:
                    phi_try, grad_try, B_try, C_try, M_try = phi_and_grad_spd(A, x_try)
            except np.linalg.LinAlgError:
                # A - C(x_try) not SPD / singular
                t *= backtracking_factor
                continue

            # Armijo: phi(x+td) <= phi(x) + alpha t g^T d
            if phi_try <= phi + armijo_alpha * t * gTd:
                success = True
                if verbose and it % 50 == 0:
                    print(f"    accepted step t={t:.3e} after {bt+1} backtracks")
                break
            else:
                t *= backtracking_factor

        if not success:
            if verbose:
                print("Backtracking failed to find a better point; stopping.")
            break

        # BFGS update on H_BFGS if using quasi-Newton
        if method == "quasi-newton":
            s = x_try - x            # step in x
            y_vec = grad_try - grad  # change in gradient
            sy = float(s @ y_vec)
            if sy > 1e-12:
                if it == 0:
                    # Optional scaling of initial inverse Hessian
                    yy = float(y_vec @ y_vec)
                    if yy > 0:
                        H_BFGS = (sy / yy) * np.eye(n - 1)
                rho = 1.0 / sy
                I = np.eye(n - 1)
                V = I - rho * np.outer(s, y_vec)
                H_BFGS = V @ H_BFGS @ V.T + rho * np.outer(s, s)
            # else: skip update if curvature condition violated

        # Accept the step
        x = x_try
        phi, grad, B, C, M = phi_try, grad_try, B_try, C_try, M_try

    return B, C, x



def spd_hilbert(n):
    """Hilbert matrix, dense SPD."""
    i = np.arange(1, n+1)
    j = np.arange(1, n+1)
    return 1.0 / (i[:, None] + j[None, :] - 1)


def spd_ar1(n, rho=0.8):
    """AR(1)-type Toeplitz covariance matrix."""
    i = np.arange(n)
    j = np.arange(n)
    return rho ** np.abs(i[:, None] - j[None, :])


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


def spd_fractional_BM(n, H=0.5, T=1.0):
    """
    Fractional Brownian motion covariance matrix on an equispaced grid.

    A[i,j] = 0.5 * (T/n)^(2H) * ( i^(2H) + j^(2H) - |i-j|^(2H) )
    with i,j = 1,...,n (1-based indices in the formula).

    Parameters
    ----------
    n : int
        Matrix size.
    H : float, default 0.5
        Hurst parameter in (0,1).
    T : float, default 1.0
        Final time horizon.
    """
    i = np.arange(1, n + 1, dtype=float)
    j = np.arange(1, n + 1, dtype=float)
    I = i[:, None]
    J = j[None, :]

    factor = 0.5 * (T / n) ** (2.0 * H)
    A = factor * (I ** (2.0 * H) + J ** (2.0 * H) - np.abs(I - J) ** (2.0 * H))
    return A



# -------- example usage --------



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    random_A = False
    np.random.seed(0)

    # Construct a random SPD matrix A
    if random_A:
        n = 20
        M = np.random.randn(n, n)
        A = M @ M.T + n * np.eye(n)  # ensures SPD
    else:
        n = 100
        A = spd_fractional_BM(n, H=0.5, T=1.0)
        # Or any other SPD constructor

    print("A =")
    print(A)

    # ---------- Gradient descent ----------
    print("\n=== Gradient Descent Method ===")
    t0 = time.perf_counter()
    B_gd, C_gd, x_gd = constrained_decomposition(
        A,
        allow_indefinite_B=False,
        tol=1e-8,
        max_iter=500,
        method="gradient-descent",
        verbose=True,
    )
    t1 = time.perf_counter()
    runtime_gd = t1 - t0
    print(f"Running time (gradient descent): {runtime_gd:.4f} seconds")

    A_rec_gd = np.linalg.inv(B_gd) + C_gd
    err_gd = np.linalg.norm(A - A_rec_gd, ord="fro")
    print("\n[GD] Reconstruction error (Frobenius):", err_gd)

    _, grad_gd, _, _, _ = phi_and_grad_spd(A, x_gd)
    grad_norm_gd = np.linalg.norm(grad_gd)
    r_gd = np.array([B_gd[k, k] - B_gd[k, k + 1] for k in range(A.shape[0] - 1)])
    max_r_gd = np.max(np.abs(r_gd))
    print(f"[GD] Final ||grad||={grad_norm_gd:.3e}, "
          f"max|Bkk-Bk,k+1|={max_r_gd:.3e}")
    print("Verification (GD): isposdef(B)=", is_spd(B_gd),
          " ; isposdef(C)=", is_spd(C_gd))

    plot_decomposition_heatmaps(A, B_gd, C_gd,
                                filename="n_" + str(n) + "_decomposition_heatmaps_gd.png")

    # ---------- Quasi-Newton (BFGS) ----------
    print("\n=== Quasi-Newton (BFGS) Method ===")
    t0 = time.perf_counter()
    B_qn, C_qn, x_qn = constrained_decomposition(
        A,
        allow_indefinite_B=False,
        tol=1e-8,
        max_iter=1000,
        method="quasi-newton",
        verbose=True,
    )
    t1 = time.perf_counter()
    runtime_qn = t1 - t0
    print(f"Running time (quasi-Newton): {runtime_qn:.4f} seconds")

    A_rec_qn = np.linalg.inv(B_qn) + C_qn
    err_qn = np.linalg.norm(A - A_rec_qn, ord="fro")
    print("\n[QN] Reconstruction error (Frobenius):", err_qn)

    _, grad_qn, _, _, _ = phi_and_grad_spd(A, x_qn)
    grad_norm_qn = np.linalg.norm(grad_qn)
    r_qn = np.array([B_qn[k, k] - B_qn[k, k + 1] for k in range(A.shape[0] - 1)])
    max_r_qn = np.max(np.abs(r_qn))
    print(f"[QN] Final ||grad||={grad_norm_qn:.3e}, "
          f"max|Bkk-Bk,k+1|={max_r_qn:.3e}")
    print("Verification (QN): isposdef(B)=", is_spd(B_qn),
          " ; isposdef(C)=", is_spd(C_qn))

    plot_decomposition_heatmaps(A, B_qn, C_qn,
                                filename="n_" + str(n) + "_decomposition_heatmaps_qn.png")

    # ---------- Newton method ----------
    print("\n=== Newton Method ===")
    t0 = time.perf_counter()
    B_newt, C_newt, x_newt = constrained_decomposition(
        A,
        allow_indefinite_B=False,
        tol=1e-8,
        max_iter=500,          # Newton should need far fewer iters
        method="newton",
        verbose=True,
    )
    t1 = time.perf_counter()
    runtime_newt = t1 - t0
    print(f"Running time (Newton): {runtime_newt:.4f} seconds")

    A_rec_newt = np.linalg.inv(B_newt) + C_newt
    err_newt = np.linalg.norm(A - A_rec_newt, ord="fro")
    print("\n[Newton] Reconstruction error (Frobenius):", err_newt)

    _, grad_newt, _, _, _ = phi_and_grad_spd(A, x_newt)
    grad_norm_newt = np.linalg.norm(grad_newt)
    r_newt = np.array([B_newt[k, k] - B_newt[k, k + 1] for k in range(A.shape[0] - 1)])
    max_r_newt = np.max(np.abs(r_newt))
    print(f"[Newton] Final ||grad||={grad_norm_newt:.3e}, "
          f"max|Bkk-Bk,k+1|={max_r_newt:.3e}")
    print("Verification (Newton): isposdef(B)=", is_spd(B_newt),
          " ; isposdef(C)=", is_spd(C_newt))

    plot_decomposition_heatmaps(A, B_newt, C_newt,
                                filename="n_" + str(n) + "_decomposition_heatmaps_newton.png")

    # ---------- Compare all three ----------
    dB_qn_newt = np.linalg.norm(B_qn - B_newt, ord="fro")
    dC_qn_newt = np.linalg.norm(C_qn - C_newt, ord="fro")
    dx_qn_newt = np.linalg.norm(x_qn - x_newt)

    print("\n=== Comparison GD vs QN vs Newton ===")
    print(f"||B_gd   - B_newt||_F = {np.linalg.norm(B_gd - B_newt, ord='fro'):.3e}")
    print(f"||C_gd   - C_newt||_F = {np.linalg.norm(C_gd - C_newt, ord='fro'):.3e}")
    print(f"||x_gd   - x_newt||   = {np.linalg.norm(x_gd - x_newt):.3e}")
    print(f"||B_qn   - B_newt||_F = {dB_qn_newt:.3e}")
    print(f"||C_qn   - C_newt||_F = {dC_qn_newt:.3e}")
    print(f"||x_qn   - x_newt||   = {dx_qn_newt:.3e}")

