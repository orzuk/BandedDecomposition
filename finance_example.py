from constrained_decomposition_core import *
from constrained_decomposition_core import make_orthogonal_complement_basis, constrained_decomposition_dual
from constrained_decomposition_matrices import *
from constrained_decomposition_viz import plot_decomposition_heatmaps
import matplotlib.pyplot as plt
from pathlib import Path
import math
import argparse
import time

# Computing value
# Markovian strategy:
def invest_value_markovian(B, C, log_flag = True):

    (signB, logabsdetB) = np.linalg.slogdet(B)
    (signC, logabsdetC) = np.linalg.slogdet(C)

    if log_flag:
        return (0.5*(logabsdetC-logabsdetB))
    else:
        return -math.exp(0.5*(logabsdetC-logabsdetB))

# General strategy:
def invest_value_general(A, log_flag = True):
    A_inv = spd_inverse(A)
    logabsdetC = -np.sum(np.log(np.diag(A_inv)))

    (signA, logabsdetA) = np.linalg.slogdet(A)

    if log_flag:
        return (0.5*(logabsdetA-logabsdetC))
    else:
        return -math.exp(0.5*(logabsdetA-logabsdetC))


def compute_value_vs_H_fbm(H_vec, n=100):
    n_H = len(H_vec)
    val_vec_markovian = np.zeros(n_H)
    val_vec_general = np.zeros(n_H)

    print(f"fBM: n={n}, matrix size={n}x{n}")
    print(f"Number of H values: {n_H}")
    total_start = time.time()

    for i in range(n_H):
        print(f"\n--- H = {H_vec[i]:.4f} ({i+1}/{n_H}) ---")

        # Build matrix
        A = spd_fractional_BM(n, H=H_vec[i], T=1.0)
        A_inv = spd_inverse(A)

        basis = TridiagC_Basis(n)  # keeps your specialized fast case
        B_newt, C_newt, x_newt = constrained_decomposition(
            A=A_inv,
            basis=basis,
            method="newton",
            tol=1e-6,
            max_iter=500,
            verbose=False
        )

        val_vec_markovian[i] = invest_value_markovian(B_newt, A)

        A_diff = spd_fractional_BM(n, H=H_vec[i], T=1.0, diff_flag=True)
        val_vec_general[i] = invest_value_general(A_diff)

        print(f"  Markovian: {val_vec_markovian[i]:.6f}, General: {val_vec_general[i]:.6f}")

    total_time = time.time() - total_start
    print(f"\n=== Total time: {total_time:.2f} seconds ({total_time/n_H:.2f} sec/H value) ===")

    return val_vec_markovian, val_vec_general


def make_mixed_fbm_full_info_basis(N: int):
    """
    Build the full-information strategy basis for mixed fBM (COO sparse format).

    S is spanned by matrices D^{k,l} in M_{2N}(R), with l = 1,...,N and k <= 2l-2:
        D^{k,l}_{ij} = 1 if i = k and j = 2l - 1
        D^{k,l}_{ij} = 1 if i = 2l - 1 and j = k
        D^{k,l}_{ij} = 0 otherwise

    Each D^{k,l} has only 2 non-zeros, so COO is very efficient.
    Dimension: O(N^2)
    """
    n = 2 * N
    coo_mats = []

    for l in range(1, N + 1):  # l = 1, ..., N
        j_col = 2 * l - 1 - 1  # 0-based: j = 2l-1 in paper -> index 2l-2
        for k in range(1, 2 * l - 1):  # k = 1, ..., 2l-2
            i_row = k - 1  # 0-based
            # Each D has 2 non-zeros: (i_row, j_col) and (j_col, i_row)
            rows = np.array([i_row, j_col], dtype=int)
            cols = np.array([j_col, i_row], dtype=int)
            vals = np.array([1.0, 1.0], dtype=float)
            coo_mats.append((rows, cols, vals))

    return SymBasis(n=n, coo_mats=coo_mats, name=f"mixed_fbm_full_info_N={N}")


def make_mixed_fbm_markovian_basis(N: int):
    """
    Build the Markovian strategy basis for mixed fBM (COO sparse format).

    S is spanned by matrices D^{l,1} and D^{l,2} for l = 1,...,N:

    D^{l,1}_{ij} = 1 if i < j, j = 2l-1, and i is odd (1-based)
                   1 if i = 2l-1, j < i, and j is odd
                   0 otherwise

    D^{l,2}_{ij} = 1 if i < j, j = 2l-1, and i is even (1-based)
                   1 if i = 2l-1, j < i, and j is even
                   0 otherwise

    Dimension: 2N = O(N)
    """
    n = 2 * N
    coo_mats = []

    for l in range(1, N + 1):  # l = 1, ..., N
        j_col = 2 * l - 1  # 1-based index for j = 2l-1

        # D^{l,1}: collect all odd i < j
        rows1, cols1, vals1 = [], [], []
        for i in range(1, n + 1):  # 1-based
            if i < j_col and (i % 2 == 1):
                # Add both (i-1, j_col-1) and (j_col-1, i-1) for symmetry
                rows1.extend([i - 1, j_col - 1])
                cols1.extend([j_col - 1, i - 1])
                vals1.extend([1.0, 1.0])
        if rows1:  # Only add if non-empty
            coo_mats.append((np.array(rows1, dtype=int), np.array(cols1, dtype=int), np.array(vals1, dtype=float)))

        # D^{l,2}: collect all even i < j
        rows2, cols2, vals2 = [], [], []
        for i in range(1, n + 1):  # 1-based
            if i < j_col and (i % 2 == 0):
                rows2.extend([i - 1, j_col - 1])
                cols2.extend([j_col - 1, i - 1])
                vals2.extend([1.0, 1.0])
        if rows2:  # Only add if non-empty
            coo_mats.append((np.array(rows2, dtype=int), np.array(cols2, dtype=int), np.array(vals2, dtype=float)))

    return SymBasis(n=n, coo_mats=coo_mats, name=f"mixed_fbm_markovian_N={N}")


def compute_value_vs_H_mixed_fbm(H_vec, N=50, alpha=1.0, delta_t=1.0, solver="primal", method="newton", strategy="both"):
    """
    Compute investment value vs Hurst parameter H for mixed fBM model.

    For each H:
      1. Build covariance matrix Σ = spd_mixed_fbm(N, H, alpha, delta_t)
      2. Compute precision Λ = Σ^{-1}
      3. For each strategy (Markovian, full-info):
         - Build the constraint subspace S
         - Decompose Λ = B^{-1} + C with B ⊥ S
         - Value = -sqrt(|B|/|Σ|), log form: 0.5*(log|Σ| - log|B|)

    Parameters
    ----------
    H_vec : array-like
        Vector of Hurst parameters to evaluate.
    N : int
        Number of time steps. Matrix size is 2N x 2N.
    alpha : float
        Weight of fBM component in mixed index.
    delta_t : float
        Time step size.
    solver : str
        "primal" uses primal Newton (fast for small m, Markovian),
                 raises max_m_for_full_hessian for full-info.
        "dual" uses dual Newton for full-info (builds S⊥ explicitly).
               Better when m is large but m⊥ = n(n+1)/2 - m is small.
    strategy : str
        "both" runs both Markovian and full-info strategies.
        "markovian" runs only Markovian strategy (fast, O(N) basis).
        "full" runs only full-info strategy (slow, O(N²) basis).

    Returns
    -------
    val_markovian : np.ndarray or None
        Log values for Markovian strategy (None if not computed).
    val_full_info : np.ndarray or None
        Log values for full-information strategy (None if not computed).
    """
    n_H = len(H_vec)
    run_markovian = strategy in ("both", "markovian")
    run_full = strategy in ("both", "full")

    val_markovian = np.zeros(n_H) if run_markovian else None
    val_full_info = np.zeros(n_H) if run_full else None

    n = 2 * N

    # Build bases as needed
    basis_markov = make_mixed_fbm_markovian_basis(N) if run_markovian else None
    basis_full = make_mixed_fbm_full_info_basis(N) if run_full else None

    print(f"Mixed fBM: N={N}, matrix size={n}x{n}")
    print(f"Strategy: {strategy}")

    if run_markovian:
        print(f"Markovian basis dimension: {basis_markov.m}")

    if run_full:
        sym_dim = n * (n + 1) // 2
        m_full = basis_full.m
        m_perp = sym_dim - m_full
        print(f"Full-info basis dimension (primal m): {m_full}")
        print(f"Full-info dual dimension (m⊥): {m_perp}")
        print(f"Solver mode: {solver}")

    # For dual solver, pre-build orthogonal complement basis (expensive but done once)
    basis_full_perp = None
    if run_full and solver == "dual":
        print("Building orthogonal complement basis for full-info (this may take a moment)...")
        t0 = time.time()
        basis_full_perp = make_orthogonal_complement_basis(basis_full)
        print(f"  Built S⊥ basis with dimension {basis_full_perp.m} in {time.time()-t0:.2f}s")

    total_start = time.time()

    for i, H in enumerate(H_vec):
        print(f"\n--- H = {H:.4f} ({i+1}/{n_H}) ---")

        # Build covariance matrix
        Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
        if not is_spd(Sigma):
            print(f"  WARNING: Sigma not SPD for H={H}")
            if run_markovian:
                val_markovian[i] = np.nan
            if run_full:
                val_full_info[i] = np.nan
            continue

        # Precision matrix (what we decompose)
        Lambda = spd_inverse(Sigma)
        if not is_spd(Lambda):
            print(f"  WARNING: Lambda not SPD for H={H}")
            if run_markovian:
                val_markovian[i] = np.nan
            if run_full:
                val_full_info[i] = np.nan
            continue

        _, log_det_Sigma = np.linalg.slogdet(Sigma)

        # --- Markovian strategy (primal, m is small) ---
        if run_markovian:
            try:
                t_markov = time.time()
                B_markov, C_markov, _, info_m = constrained_decomposition(
                    A=Lambda,
                    basis=basis_markov,
                    method=method,
                    tol=1e-8,
                    max_iter=500,
                    verbose=False,
                    return_info=True,
                )
                t_markov = time.time() - t_markov
                if i == 0:
                    print(f"  [Markovian using: {info_m.get('used_method', 'unknown')}]")
                _, log_det_B_markov = np.linalg.slogdet(B_markov)
                val_markovian[i] = 0.5 * (log_det_Sigma - log_det_B_markov)
                print(f"  Markovian: log(value) = {val_markovian[i]:.6f} ({t_markov:.2f}s, {info_m['iters']} iters)")
            except Exception as e:
                print(f"  Markovian FAILED: {e}")
                val_markovian[i] = np.nan

        # --- Full-information strategy ---
        if run_full:
            try:
                t_full = time.time()
                if solver == "dual":
                    # Dual Newton: optimize over S⊥ directly
                    B_full, C_full, _, _ = constrained_decomposition_dual(
                        A=Lambda,
                        basis=basis_full,
                        basis_perp=basis_full_perp,
                        tol=1e-8,
                        max_iter=500,
                        verbose=False
                    )
                    info = {"iters": "?", "used_method": "dual"}
                else:
                    # Primal with specified method (auto-switches to newton-cg for large m)
                    B_full, C_full, _, info = constrained_decomposition(
                        A=Lambda,
                        basis=basis_full,
                        method=method,
                        tol=1e-8,
                        max_iter=500,
                        verbose=False,
                        return_info=True,
                    )
                    if i == 0:  # Print method used on first H
                        print(f"  [Full-info using: {info.get('used_method', 'unknown')}]")
                t_full = time.time() - t_full

                _, log_det_B_full = np.linalg.slogdet(B_full)
                val_full_info[i] = 0.5 * (log_det_Sigma - log_det_B_full)
                print(f"  Full-info: log(value) = {val_full_info[i]:.6f} ({t_full:.2f}s, {info['iters']} iters)")
            except Exception as e:
                print(f"  Full-info FAILED: {e}")
                val_full_info[i] = np.nan

    total_time = time.time() - total_start
    print(f"\n=== Total time: {total_time:.2f} seconds ({total_time/n_H:.2f} sec/H value) ===")

    return val_markovian, val_full_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finance example: fBM or mixed fBM")
    parser.add_argument("--model", type=str, choices=["fbm", "mixed_fbm"], default="mixed_fbm",
                        help="Model type: 'fbm' or 'mixed_fbm' (default: mixed_fbm)")
    parser.add_argument("--n", type=int, default=100,
                        help="Matrix dimension (default: 100). For mixed_fbm, N=n//2 time steps.")
    parser.add_argument("--solver", type=str, choices=["primal", "dual"], default="primal",
                        help="Solver for full-info: 'primal' or 'dual' (Newton on S⊥)")
    parser.add_argument("--method", type=str, choices=["newton", "newton-cg", "quasi-newton"], default="newton",
                        help="Optimization method: 'newton' (auto-switches to newton-cg for large m), "
                             "'newton-cg' (matrix-free), or 'quasi-newton' (BFGS)")
    parser.add_argument("--strategy", type=str, choices=["both", "markovian", "full"], default="both",
                        help="Which strategies to run: 'both', 'markovian' (fast, O(N)), or 'full' (slow, O(N²))")
    args = parser.parse_args()

    model_type = args.model
    n = args.n
    solver = args.solver
    method = args.method
    strategy = args.strategy
    print(f"\n{'='*60}")
    print(f"Running model: {model_type}, n={n}, solver={solver}, method={method}, strategy={strategy}")
    print(f"{'='*60}\n")

    # --- Experiment settings ---
    H_vec = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9 with step 0.1 (coarse for profiling)

    if model_type == "fbm":
        pass  # n is used directly
    else:  # mixed_fbm
        N = n // 2  # Number of time steps (matrix is 2N x 2N)
        alpha = 1.0
        delta_t = 1.0

    # --- Run ---
    if model_type == "fbm":
        val_markov, val_general = compute_value_vs_H_fbm(H_vec, n=n)
    else:  # mixed_fbm
        val_markov, val_general = compute_value_vs_H_mixed_fbm(
            H_vec, N=N, alpha=alpha, delta_t=delta_t, solver=solver, method=method, strategy=strategy
        )

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    if val_markov is not None:
        plt.plot(H_vec, val_markov, 'b-o', label="Markovian strategy", markersize=4)
    if val_general is not None:
        plt.plot(H_vec, val_general, 'r-s', label="Full-information strategy", markersize=4)
    plt.xlabel("Hurst parameter H", fontsize=12)
    plt.ylabel("log(Value)", fontsize=12)
    if model_type == "mixed_fbm":
        plt.title(f"Mixed fBM: Strategy value vs H (N={N}, α={alpha})", fontsize=13)
        plt.axvline(x=0.75, color='gray', linestyle='--', alpha=0.5, label='H=3/4 (arbitrage-free boundary)')
    else:
        plt.title(f"fBM: Strategy value vs H (n={n})", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Save in figs/ next to this script ---
    here = Path(__file__).resolve().parent
    fig_dir = here / "figs" / model_type
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_png = fig_dir / f"value_{model_type}_vs_H_n_{n}_{strategy}.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved value figure to: {out_png}")


    # ---- Save decomposition heatmaps for a chosen H ----
    H0 = 0.8  # pick one (or loop over a few)

    if model_type == "fbm":
        A = spd_fractional_BM(n, H=H0, T=1.0, diff_flag=True)
        A_inv = spd_inverse(A)
        basis = TridiagC_Basis(n)
    else:  # mixed_fbm
        Sigma = spd_mixed_fbm(N, H=H0, alpha=alpha, delta_t=delta_t)
        A_inv = spd_inverse(Sigma)
        basis = make_mixed_fbm_markovian_basis(N)  # Use Markovian for heatmap

    B0, C0, x0 = constrained_decomposition(
        A=A_inv,
        basis=basis,
        method="newton",
        tol=1e-6,
        max_iter=500,
        verbose=False
    )

    out_heat = fig_dir / f"heatmap_{model_type}_H_{H0:.2f}_n_{n}.png"
    plot_decomposition_heatmaps(
        A=A_inv,
        B=B0,
        C=C0,
        basis=basis,
        out_file=out_heat,
    )
    print(f"Saved heatmap to: {out_heat}")
