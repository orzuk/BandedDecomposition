from constrained_decomposition_core import *
from constrained_decomposition_core import make_orthogonal_complement_basis, constrained_decomposition_dual
from constrained_decomposition_matrices import *
from constrained_decomposition_viz import plot_decomposition_heatmaps
import matplotlib.pyplot as plt
from pathlib import Path
import math
import argparse
import time
import multiprocessing as mp
import os

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


def invest_value_mixed_fbm(H, N, alpha, delta_t, strategy, method="newton", solver="primal",
                           Sigma=None, Lambda=None, basis=None, basis_perp=None,
                           tol=1e-8, max_iter=500, verbose=False, x_init=None):
    """
    Compute investment value for mixed fBM model with a given strategy.

    This is the unified value computation function for all three strategies:
      - "sum": Observes only the mixed index X (N observations, N×N matrix)
      - "markovian": Observes both X and W with Markovian constraints (2N obs, 2N×2N matrix)
      - "full": Observes both X and W with full information (2N obs, 2N×2N matrix)

    Value formula: log(Value) = 0.5 × (log|Σ| - log|B|)

    The only difference between strategies is how B is computed:
      - "sum": B is diagonal with B_ii = 1/Λ_ii (closed-form solution)
      - "markovian"/"full": B comes from constrained decomposition Λ = B^{-1} + C

    Parameters
    ----------
    H : float
        Hurst parameter.
    N : int
        Number of time steps.
    alpha : float
        Weight of fBM component.
    delta_t : float
        Time step size.
    strategy : str
        "sum", "markovian", or "full".
    method : str
        Optimization method for decomposition ("newton", "newton-cg", "quasi-newton").
    solver : str
        "primal" or "dual" (for full strategy only).
    Sigma : np.ndarray, optional
        Pre-computed covariance matrix. Built if not provided.
    Lambda : np.ndarray, optional
        Pre-computed precision matrix. Built if not provided.
    basis : SymBasis, optional
        Pre-computed basis for the strategy. Built if not provided.
    basis_perp : SymBasis, optional
        Pre-computed orthogonal complement basis (for dual solver).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    verbose : bool
        Print solver progress.

    Returns
    -------
    value : float
        Log investment value.
    info : dict
        Additional info: {"iters": int, "time": float, "method": str, "error": str or None}
    """
    info = {"iters": 0, "time": 0.0, "method": strategy, "error": None, "x": None}
    t_start = time.time()

    try:
        # === Step 1: Build covariance matrix Sigma (N×N for sum, 2N×2N for others) ===
        if Sigma is None:
            if strategy == "sum":
                Sigma = spd_sum_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
            else:
                Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)

        if not is_spd(Sigma):
            info["error"] = "Sigma not SPD"
            return np.nan, info

        # === Step 2: Compute precision matrix Lambda ===
        if Lambda is None:
            Lambda = spd_inverse(Sigma)

        if not is_spd(Lambda):
            info["error"] = "Lambda not SPD"
            return np.nan, info

        # === Step 3: Compute log|Σ| (shared across all strategies) ===
        _, log_det_Sigma = np.linalg.slogdet(Sigma)

        # === Step 4: Compute B (strategy-specific) ===
        if strategy == "sum":
            # Closed-form: optimal diagonal B has B_ii = 1/Λ_ii
            B = np.diag(1.0 / np.diag(Lambda))
            info["method"] = "diagonal B"
        else:
            # Decomposition: solve Λ = B^{-1} + C with C ⊥ S
            if basis is None:
                if strategy == "markovian":
                    basis = make_mixed_fbm_markovian_basis(N)
                elif strategy == "full":
                    basis = make_mixed_fbm_full_info_basis(N)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

            if strategy == "full" and solver == "dual":
                if basis_perp is None:
                    basis_perp = make_orthogonal_complement_basis(basis)
                B, _, _, _ = constrained_decomposition_dual(
                    A=Lambda, basis=basis, basis_perp=basis_perp,
                    tol=tol, max_iter=max_iter, verbose=verbose
                )
                info["iters"] = "?"
                info["method"] = "dual"
            else:
                B, _, x, decomp_info = constrained_decomposition(
                    A=Lambda, basis=basis, method=method,
                    tol=tol, max_iter=max_iter, verbose=verbose, return_info=True,
                    x_init=x_init
                )
                info["iters"] = decomp_info["iters"]
                info["method"] = decomp_info.get("used_method", method)
                info["x"] = x  # Return for warm starting next iteration

        # === Step 5: Compute log|B| and final value (shared formula) ===
        _, log_det_B = np.linalg.slogdet(B)
        value = 0.5 * (log_det_Sigma - log_det_B)

    except Exception as e:
        info["error"] = str(e)
        value = np.nan

    info["time"] = time.time() - t_start
    return value, info


def compute_value_vs_H_mixed_fbm(H_vec, N=50, alpha=1.0, delta_t=None, solver="primal", method="newton", strategy="both"):
    """
    Compute investment value vs Hurst parameter H for mixed fBM model.

    Parameters
    ----------
    H_vec : array-like
        Vector of Hurst parameters to evaluate.
    N : int
        Number of time steps. Matrix size is 2N x 2N.
    alpha : float
        Weight of fBM component in mixed index.
    delta_t : float, optional
        Time step size. Defaults to 1/N.
    solver : str
        "primal" or "dual" (for full strategy).
    method : str
        Optimization method ("newton", "newton-cg", "quasi-newton").
    strategy : str
        "both", "markovian", "full", or "sum".

    Returns
    -------
    val_markovian : np.ndarray or None
    val_full_info : np.ndarray or None
    val_sum_fbm : np.ndarray
    """
    if delta_t is None:
        delta_t = 1.0 / N

    n_H = len(H_vec)
    run_sum = True  # Always compute sum for comparison
    run_markovian = strategy in ("both", "markovian")
    run_full = strategy in ("both", "full")

    val_sum_fbm = np.zeros(n_H)
    val_markovian = np.zeros(n_H) if run_markovian else None
    val_full_info = np.zeros(n_H) if run_full else None

    n = 2 * N

    # Pre-build bases for efficiency (reused across H values)
    basis_markov = make_mixed_fbm_markovian_basis(N) if run_markovian else None
    basis_full = make_mixed_fbm_full_info_basis(N) if run_full else None
    basis_full_perp = None

    print(f"Mixed fBM: N={N}, matrix size={n}x{n}, delta_t={delta_t:.6f}")
    print(f"Strategy: {strategy}")

    if run_markovian:
        print(f"Markovian basis dimension: {basis_markov.m}")
    if run_full:
        print(f"Full-info basis dimension: {basis_full.m}")
        if solver == "dual":
            print("Building orthogonal complement basis...")
            t0 = time.time()
            basis_full_perp = make_orthogonal_complement_basis(basis_full)
            print(f"  Built S⊥ (dim={basis_full_perp.m}) in {time.time()-t0:.2f}s")

    total_start = time.time()

    # Warm start: keep track of previous solutions
    x_markov_prev = None
    x_full_prev = None

    for i, H in enumerate(H_vec):
        print(f"\n--- H = {H:.4f} ({i+1}/{n_H}) ---")

        # Build shared Sigma/Lambda for markovian/full (sum builds its own)
        Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
        Lambda = spd_inverse(Sigma) if is_spd(Sigma) else None

        # --- Sum strategy ---
        val, info = invest_value_mixed_fbm(
            H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="sum"
        )
        val_sum_fbm[i] = val
        if info["error"]:
            print(f"  Sum: FAILED - {info['error']}")
        else:
            print(f"  Sum: {val:.6f} ({info['time']:.2f}s)")

        # --- Markovian strategy ---
        if run_markovian:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="markovian",
                method=method, Sigma=Sigma, Lambda=Lambda, basis=basis_markov,
                tol=5e-8, x_init=x_markov_prev
            )
            val_markovian[i] = val
            x_markov_prev = info.get("x")  # Update warm start for next H
            if info["error"]:
                print(f"  Markovian: FAILED - {info['error']}")
            else:
                print(f"  Markovian: {val:.6f} ({info['time']:.2f}s, {info['iters']} iters, {info['method']})")

        # --- Full strategy ---
        if run_full:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="full",
                method=method, solver=solver, Sigma=Sigma, Lambda=Lambda,
                basis=basis_full, basis_perp=basis_full_perp, x_init=x_full_prev
            )
            val_full_info[i] = val
            x_full_prev = info.get("x")  # Update warm start for next H
            if info["error"]:
                print(f"  Full-info: FAILED - {info['error']}")
            else:
                print(f"  Full-info: {val:.6f} ({info['time']:.2f}s, {info['iters']} iters, {info['method']})")

    total_time = time.time() - total_start
    print(f"\n=== Total time: {total_time:.2f} seconds ({total_time/n_H:.2f} sec/H value) ===")

    return val_markovian, val_full_info, val_sum_fbm


def _compute_single_H(args):
    """Worker function for parallel H computation."""
    H, N, alpha, delta_t, method, strategy, basis_markov_data, basis_full_data = args

    run_markovian = strategy in ("both", "markovian")
    run_full = strategy in ("both", "full")

    result = {"H": H, "val_markovian": np.nan, "val_full_info": np.nan, "val_sum_fbm": np.nan,
              "iters_markov": 0, "iters_full": 0, "time": 0, "error": None,
              "time_sum": 0, "time_markov": 0, "time_full": 0}

    t_start = time.time()

    try:
        # Rebuild bases from pre-generated data
        n_basis = 2 * N
        basis_markov = SymBasis(n=n_basis, coo_mats=basis_markov_data, name=f"mixed_fbm_markovian_N={N}") if run_markovian else None
        basis_full = SymBasis(n=n_basis, coo_mats=basis_full_data, name=f"mixed_fbm_full_info_N={N}") if run_full else None

        # Build shared Sigma/Lambda
        Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
        Lambda = spd_inverse(Sigma) if is_spd(Sigma) else None

        # Sum strategy
        val, info = invest_value_mixed_fbm(H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="sum")
        result["val_sum_fbm"] = val
        result["time_sum"] = info["time"]
        if info["error"]:
            print(f"  [H={H:.4f}] Sum FAILED: {info['error']}")

        # Markovian strategy
        if run_markovian:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="markovian",
                method="newton", Sigma=Sigma, Lambda=Lambda, basis=basis_markov, tol=1e-6
            )
            result["val_markovian"] = val
            result["iters_markov"] = info["iters"]
            result["time_markov"] = info["time"]
            if info["error"]:
                result["error"] = info["error"]

        # Full-info strategy
        if run_full:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="full",
                method="newton-cg", Sigma=Sigma, Lambda=Lambda, basis=basis_full
            )
            result["val_full_info"] = val
            result["iters_full"] = info["iters"]
            result["time_full"] = info["time"]
            if info["error"]:
                result["error"] = info["error"]

    except Exception as e:
        result["error"] = str(e)

    result["time"] = time.time() - t_start
    return result


def compute_value_vs_H_mixed_fbm_parallel(H_vec, N=50, alpha=1.0, delta_t=None,
                                          method="newton", strategy="markovian", workers=None):
    """
    Parallel version of compute_value_vs_H_mixed_fbm.

    Parameters
    ----------
    workers : int or None
        Number of worker processes. Default: cpu_count - 2
    """
    if delta_t is None:
        delta_t = 1.0 / N

    n_H = len(H_vec)
    run_markovian = strategy in ("both", "markovian")
    run_full = strategy in ("both", "full")

    if workers is None:
        workers = max(1, os.cpu_count() - 2)

    n = 2 * N
    print(f"Mixed fBM (PARALLEL): N={N}, matrix size={n}x{n}, delta_t={delta_t:.6f}")
    print(f"Strategy: {strategy}, Workers: {workers}")
    print(f"H values: {n_H} (from {H_vec[0]:.4f} to {H_vec[-1]:.4f})")

    # --- Pre-generate basis data to avoid re-computing in every worker ---
    print("Pre-generating basis data for parallel workers...")
    t_basis_start = time.time()
    
    basis_markov_coo_data = None
    if run_markovian:
        markov_coo_mats = []
        for l in range(1, N + 1):
            j_col = 2 * l - 1
            rows1, cols1, vals1 = [], [], []
            for i in range(1, n + 1):
                if i < j_col and (i % 2 == 1):
                    rows1.extend([i - 1, j_col - 1])
                    cols1.extend([j_col - 1, i - 1])
                    vals1.extend([1.0, 1.0])
            if rows1:
                markov_coo_mats.append((np.array(rows1, dtype=int), np.array(cols1, dtype=int), np.array(vals1, dtype=float)))
            
            rows2, cols2, vals2 = [], [], []
            for i in range(1, n + 1):
                if i < j_col and (i % 2 == 0):
                    rows2.extend([i - 1, j_col - 1])
                    cols2.extend([j_col - 1, i - 1])
                    vals2.extend([1.0, 1.0])
            if rows2:
                markov_coo_mats.append((np.array(rows2, dtype=int), np.array(cols2, dtype=int), np.array(vals2, dtype=float)))
        basis_markov_coo_data = markov_coo_mats

    basis_full_coo_data = None
    if run_full:
        full_coo_mats = []
        for l in range(1, N + 1):
            j_col = 2 * l - 1 - 1
            for k in range(1, 2 * l - 1):
                i_row = k - 1
                rows = np.array([i_row, j_col], dtype=int)
                cols = np.array([j_col, i_row], dtype=int)
                vals = np.array([1.0, 1.0], dtype=float)
                full_coo_mats.append((rows, cols, vals))
        basis_full_coo_data = full_coo_mats
        
    print(f"... basis data generated in {time.time() - t_basis_start:.2f}s")
    print()  # blank line before results

    # Prepare arguments for workers (basis rebuilt in each worker)
    args_list = [(H, N, alpha, delta_t, method, strategy, basis_markov_coo_data, basis_full_coo_data) for H in H_vec]

    total_start = time.time()

    # Store results by H value for correct ordering at end
    results_dict = {}
    completed = 0

    # Run in parallel with imap_unordered for streaming results
    with mp.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(_compute_single_H, args_list):
            completed += 1
            results_dict[res["H"]] = res

            # Print progress immediately as each H completes
            if res["error"]:
                print(f"  [{completed:3d}/{n_H}] H={res['H']:.4f}: ERROR - {res['error']}")
            else:
                info_str = []
                info_str.append(f"sum={res['val_sum_fbm']:.6f} [{res['time_sum']:.1f}s]")
                if run_markovian:
                    info_str.append(f"markov={res['val_markovian']:.6f} ({res['iters_markov']} it) [{res['time_markov']:.1f}s]")
                if run_full:
                    info_str.append(f"full={res['val_full_info']:.6f} ({res['iters_full']} it) [{res['time_full']:.1f}s]")
                print(f"  [{completed:3d}/{n_H}] H={res['H']:.4f}: {', '.join(info_str)} [total={res['time']:.1f}s]", flush=True)

    # Collect results in original H order
    val_markovian = np.zeros(n_H) if run_markovian else None
    val_full_info = np.zeros(n_H) if run_full else None
    val_sum_fbm = np.zeros(n_H)

    for i, H in enumerate(H_vec):
        res = results_dict[H]
        if run_markovian:
            val_markovian[i] = res["val_markovian"]
        if run_full:
            val_full_info[i] = res["val_full_info"]
        val_sum_fbm[i] = res["val_sum_fbm"]

    total_time = time.time() - total_start
    print(f"\n=== Total time: {total_time:.2f} seconds ({total_time/n_H:.2f} sec/H value) ===")
    print(f"=== Parallel speedup: {workers}x theoretical, actual depends on load balance ===")

    return val_markovian, val_full_info, val_sum_fbm


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
    parser.add_argument("--hres", type=float, default=0.1,
                        help="H resolution step size (default: 0.1). E.g., 0.1 gives H=0.1,0.2,...,0.9; "
                             "0.02 gives H=0.02,0.04,...,0.98")
    parser.add_argument("--hmin", type=float, default=0.0,
                        help="Minimum H value (default: 0.0). H range starts at max(hmin, hres).")
    parser.add_argument("--hmax", type=float, default=1.0,
                        help="Maximum H value exclusive (default: 1.0). H range ends before hmax.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight of fBM component in mixed index (default: 1.0). "
                             "Higher alpha increases fBM influence vs BM.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run H values in parallel using multiprocessing")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 2)")
    args = parser.parse_args()

    model_type = args.model
    n = args.n
    solver = args.solver
    method = args.method
    strategy = args.strategy
    hres = args.hres
    hmin = args.hmin
    hmax = args.hmax
    alpha = args.alpha
    parallel = args.parallel
    workers = args.workers

    if workers is None:
        workers = max(1, os.cpu_count() - 2)

    print(f"\n{'='*60}")
    print(f"Running model: {model_type}, n={n}, solver={solver}, method={method}, strategy={strategy}")
    print(f"H range: [{hmin}, {hmax}) with step {hres}, alpha={alpha}")
    if parallel:
        print(f"PARALLEL mode: {workers} workers")
    print(f"{'='*60}\n")

    # --- Experiment settings ---
    # H range: from max(hmin, hres) to hmax (exclusive), with step hres
    h_start = max(hmin, hres) if hmin == 0.0 else hmin
    H_vec = np.arange(h_start, hmax, hres)

    if model_type == "fbm":
        pass  # n is used directly
    else:  # mixed_fbm
        N = n // 2  # Number of time steps (matrix is 2N x 2N)
        delta_t = 1.0 / N  # Time step for consistent scaling between sum and mixed

    # --- Run ---
    if model_type == "fbm":
        val_markov, val_general = compute_value_vs_H_fbm(H_vec, n=n)
        val_sum = None
    else:  # mixed_fbm
        if parallel:
            val_markov, val_general, val_sum = compute_value_vs_H_mixed_fbm_parallel(
                H_vec, N=N, alpha=alpha, delta_t=delta_t, method=method, strategy=strategy, workers=workers
            )
        else:
            val_markov, val_general, val_sum = compute_value_vs_H_mixed_fbm(
                H_vec, N=N, alpha=alpha, delta_t=delta_t, solver=solver, method=method, strategy=strategy
            )

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    if val_markov is not None:
        plt.plot(H_vec, val_markov, 'b-o', label="Markovian strategy", markersize=4)
    if val_general is not None:
        plt.plot(H_vec, val_general, 'r-s', label="Full-information strategy", markersize=4)
    if val_sum is not None:
        plt.plot(H_vec, val_sum, 'g-^', label="Sum strategy (no decomp)", markersize=4)
    plt.xlabel("Hurst parameter H", fontsize=12)
    plt.ylabel("log(Value)", fontsize=12)
    if model_type == "mixed_fbm":
        plt.title(f"Mixed fBM: Strategy value vs H (N={N}, α={alpha})", fontsize=13)
        # Only show H=3/4 line if it's in the plotted range
        if H_vec[0] <= 0.75 <= H_vec[-1]:
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

    # Include H range and alpha in filename for uniqueness
    h_range_str = f"H_{H_vec[0]:.2f}_{H_vec[-1]:.2f}"
    alpha_str = f"a{alpha:.1f}" if alpha != 1.0 else ""
    out_png = fig_dir / f"value_{model_type}_n_{n}_{h_range_str}{alpha_str}_{strategy}.png"
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
