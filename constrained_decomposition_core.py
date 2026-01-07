"""
Core algorithms and data structures for constrained SPD decompositions.

This module contains:
  - Symmetric subspace bases (dense/sparse/circulant/tridiagonal parameterizations)
  - Primal and dual solvers (Newton / quasi-Newton / gradient)
  - Group-invariant and circulant specializations
  - Core linear-algebra helpers (SPD checks, inverse, etc.)

It is intentionally free of demo/CLI/plotting code.
"""


import numpy as np
import time
from scipy import sparse
from scipy import linalg as sp_linalg

def block_reynolds_project(A: np.ndarray, blocks):
    """
    Reynolds projection for the within-block permutation group:
      - for bi!=bj: entries become constant on each (block_i, block_j) rectangle
      - for bi==bj: diagonals become constant (mean of diagonal),
                    off-diagonals become constant (mean of off-diagonal)
    This equals (1/|G|) sum_{P in G} P A P^T for the product of within-block permutations,
    hence preserves SPD.
    """
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    out = np.zeros_like(A)

    # normalize blocks to 1D int arrays
    blk = [np.asarray(I, dtype=int).reshape(-1) for I in blocks]

    for bi, I in enumerate(blk):
        # within-block part (diagonal vs off-diagonal treated separately)
        sub = A[np.ix_(I, I)]
        if len(I) == 1:
            out[np.ix_(I, I)] = sub
        else:
            diag_mean = float(np.mean(np.diag(sub)))
            mask = ~np.eye(len(I), dtype=bool)
            off_mean = float(np.mean(sub[mask])) if np.any(mask) else 0.0
            tmp = np.full((len(I), len(I)), off_mean, dtype=float)
            np.fill_diagonal(tmp, diag_mean)
            out[np.ix_(I, I)] = tmp

        # cross-block rectangles
        for bj in range(bi + 1, len(blk)):
            J = blk[bj]
            m = float(np.mean(A[np.ix_(I, J)]))
            out[np.ix_(I, J)] = m
            out[np.ix_(J, I)] = m

    return 0.5 * (out + out.T)



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


def _logdet_spd_from_cholesky(L):
    return 2.0 * np.sum(np.log(np.diag(L)))


class SymBasis:
    """
    Represents a symmetric matrix subspace S = span(D1,...,Dm).

    Supports:
      - dense basis matrices (list of dense Dk)
      - sparse basis matrices in COO format: (rows, cols, vals) per Dk (0-based)

    Stage-1/implicit-B helpers:
      - cached index sets needed for trace computations
      - fast linear combination D(v) in COO (for Hv)
    """

    def __init__(self, n, dense_mats=None, coo_mats=None, name="generic"):
        self.n = int(n)
        self.name = name

        if dense_mats is None and coo_mats is None:
            raise ValueError("Provide either dense_mats or coo_mats.")
        if dense_mats is not None and coo_mats is not None:
            raise ValueError("Provide only one of dense_mats or coo_mats.")

        self._dense = None
        self._coo = None

        if dense_mats is not None:
            mats = [np.asarray(D, dtype=float) for D in dense_mats]
            for D in mats:
                if D.shape != (self.n, self.n):
                    raise ValueError("All Dk must be (n,n).")
                if not np.allclose(D, D.T, atol=1e-12):
                    raise ValueError("All Dk must be symmetric.")
            self._dense = mats

        if coo_mats is not None:
            parsed = []
            for item in coo_mats:
                rows, cols, vals = item
                rows = np.asarray(rows, dtype=int)
                cols = np.asarray(cols, dtype=int)
                vals = np.asarray(vals, dtype=float)
                if rows.shape != cols.shape or rows.shape != vals.shape:
                    raise ValueError("COO arrays must have same shape.")
                if np.any(rows < 0) or np.any(rows >= self.n) or np.any(cols < 0) or np.any(cols >= self.n):
                    raise ValueError("COO indices out of range.")
                parsed.append((rows, cols, vals))
            self._coo = parsed

        self.m = len(self._dense) if self._dense is not None else len(self._coo)

        # caches for COO
        self._cached_cols = None     # columns needed for gradient traces
        self._cached_I = None        # indices (rows∪cols) needed for Hv traces
        self._cached_col_pos = None  # map: col -> position in cached_cols
        self._cached_I_pos = None    # map: idx -> position in cached_I

        # Precompute concatenated COO arrays for fast coo_linear_combo
        if self._coo is not None:
            self._precompute_concat_coo()

    def is_sparse_coo(self) -> bool:
        return self._coo is not None

    def _precompute_concat_coo(self):
        """
        Precompute concatenated COO arrays for fast coo_linear_combo.

        Creates:
          _all_rows, _all_cols, _all_vals: concatenated COO data
          _basis_idx: which basis matrix each entry belongs to
          _nnz_cumsum: cumulative nnz for slicing
        """
        if self._coo is None:
            return

        total_nnz = sum(len(r) for r, c, v in self._coo)
        all_rows = np.empty(total_nnz, dtype=np.int32)
        all_cols = np.empty(total_nnz, dtype=np.int32)
        all_vals = np.empty(total_nnz, dtype=np.float64)
        basis_idx = np.empty(total_nnz, dtype=np.int32)

        offset = 0
        for l, (rows, cols, vals) in enumerate(self._coo):
            k = len(rows)
            all_rows[offset:offset + k] = rows
            all_cols[offset:offset + k] = cols
            all_vals[offset:offset + k] = vals
            basis_idx[offset:offset + k] = l
            offset += k

        self._all_rows = all_rows
        self._all_cols = all_cols
        self._all_vals = all_vals
        self._basis_idx = basis_idx

    # -----------------------------
    # Default linear map C(x)
    # -----------------------------
    def build_C(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.m,):
            raise ValueError(f"x must have shape ({self.m},)")
        C = np.zeros((self.n, self.n), dtype=float)

        if self._dense is not None:
            for k, Dk in enumerate(self._dense):
                C += x[k] * Dk
            return C

        for k, (rows, cols, vals) in enumerate(self._coo):
            C[rows, cols] += x[k] * vals
        return C

    # -----------------------------
    # Dense B tracer
    # -----------------------------
    def trace_with(self, B):
        B = np.asarray(B, dtype=float)
        if B.shape != (self.n, self.n):
            raise ValueError("B has wrong shape.")
        g = np.zeros(self.m, dtype=float)

        if self._dense is not None:
            for k, Dk in enumerate(self._dense):
                g[k] = float(np.sum(B * Dk))
            return g

        for k, (rows, cols, vals) in enumerate(self._coo):
            g[k] = float(np.sum(B[rows, cols] * vals))
        return g

    # -----------------------------
    # Hessian computation
    # -----------------------------
    def hessian_from_B(self, B):
        """
        Optional: override in subclass to compute H_{k,l} = tr(B Dk B Dl) faster.
        Returns None by default (caller should use generic_hessian_from_B).
        """
        return None

    def generic_hessian_from_B(self, B, max_m_for_full=1000):
        """
        Generic Hessian H_{k,l} = tr(B Dk B Dl).
        """
        m = self.m
        n = self.n
        if m > max_m_for_full:
            raise ValueError(
                f"m={m} is large; forming the full Hessian is expensive. "
                f"Use newton-cg or quasi-newton instead."
            )

        B = np.asarray(B, dtype=float)

        if self._dense is not None:
            BD = np.stack([B @ Dk for Dk in self._dense], axis=0)  # (m,n,n)
        else:
            BD = np.zeros((m, n, n), dtype=float)
            for k, (rows, cols, vals) in enumerate(self._coo):
                for i, j, v in zip(rows, cols, vals):
                    BD[k, :, j] += v * B[:, i]

        H = np.einsum("kij,lji->kl", BD, BD, optimize=True)
        H = 0.5 * (H + H.T)
        return H

    # -----------------------------
    # Bandwidth detection for banded optimization
    # -----------------------------
    def max_bandwidth(self):
        """
        Compute the maximum bandwidth of all basis matrices.

        Returns the maximum |i - j| over all non-zero entries in all Dk.
        If all Dk have bandwidth <= b and A has bandwidth <= b,
        then M = A - C(x) has bandwidth <= b, enabling O(n·b²) Cholesky.
        """
        if self._coo is not None:
            max_bw = 0
            for rows, cols, vals in self._coo:
                if len(rows) > 0:
                    bw = np.max(np.abs(rows - cols))
                    max_bw = max(max_bw, bw)
            return int(max_bw)
        elif self._dense is not None:
            max_bw = 0
            for Dk in self._dense:
                bw_l, bw_u = detect_bandwidth(Dk)
                max_bw = max(max_bw, bw_l, bw_u)
            return max_bw
        return 0

    # -----------------------------
    # COO caches
    # -----------------------------
    def required_columns_for_traces(self, rebuild_cache=False):
        """
        For gradient g_k = sum_{(i,j) in supp(Dk)} Dk_ij * B_ij,
        we need B_ij entries, so we need the columns j that appear in any Dk.
        """
        if self._coo is None:
            return None
        if self._cached_cols is not None and not rebuild_cache:
            return self._cached_cols

        cols = np.unique(np.concatenate([cols for (rows, cols, vals) in self._coo])).astype(int)
        self._cached_cols = cols
        self._cached_col_pos = {int(c): t for t, c in enumerate(cols)}
        # Array version for vectorized lookup
        self._cached_col_pos_arr = np.full(self.n, -1, dtype=np.int32)
        self._cached_col_pos_arr[cols] = np.arange(len(cols), dtype=np.int32)
        return cols

    def required_indices_rows_cols(self, rebuild_cache=False):
        """
        For Hv in the COO implicit method we will build a small r×r matrix
        G = Z^T D(v) Z where Z contains columns of B indexed by I = union(rows, cols).
        """
        if self._coo is None:
            return None
        if self._cached_I is not None and not rebuild_cache:
            return self._cached_I

        all_rows = np.concatenate([rows for (rows, cols, vals) in self._coo])
        all_cols = np.concatenate([cols for (rows, cols, vals) in self._coo])
        I = np.unique(np.concatenate([all_rows, all_cols])).astype(int)
        self._cached_I = I
        self._cached_I_pos = {int(i): t for t, i in enumerate(I)}
        # Array version for vectorized lookup
        self._cached_I_pos_arr = np.full(self.n, -1, dtype=np.int32)
        self._cached_I_pos_arr[I] = np.arange(len(I), dtype=np.int32)
        return I

    # -----------------------------
    # COO trace from selected columns
    # -----------------------------
    def trace_from_selected_cols(self, Z, col_pos_arr=None):
        """
        Compute g_k = tr(B Dk) from Z = B[:, J] where J = required_columns_for_traces().

        Vectorized: uses precomputed concatenated COO arrays.
        """
        if self._coo is None:
            raise ValueError("trace_from_selected_cols requires COO basis.")
        if col_pos_arr is None:
            if self._cached_col_pos_arr is None:
                self.required_columns_for_traces(rebuild_cache=True)
            col_pos_arr = self._cached_col_pos_arr

        # Vectorized: look up all (row, col_pos[col]) entries at once
        row_idx = self._all_rows
        col_idx = col_pos_arr[self._all_cols]
        products = self._all_vals * Z[row_idx, col_idx]

        # Sum by basis index using bincount
        g = np.bincount(self._basis_idx, weights=products, minlength=self.m)
        return g

    def trace_from_small_G(self, G, I_pos=None, I_pos_arr=None):
        """
        Compute h_k = sum_{(i,j)} Dk_ij * W_ij where W restricted to I×I is represented by G.
        Here G[p,q] = W[I[p], I[q]].

        Vectorized: uses precomputed concatenated COO arrays.
        """
        if self._coo is None:
            raise ValueError("trace_from_small_G requires COO basis.")
        if I_pos_arr is None:
            if self._cached_I_pos_arr is None:
                self.required_indices_rows_cols(rebuild_cache=True)
            I_pos_arr = self._cached_I_pos_arr

        # Vectorized: look up all G[I_pos[row], I_pos[col]] at once
        row_idx = I_pos_arr[self._all_rows]
        col_idx = I_pos_arr[self._all_cols]
        products = self._all_vals * G[row_idx, col_idx]

        # Sum by basis index using bincount
        h = np.bincount(self._basis_idx, weights=products, minlength=self.m)
        return h

    # -----------------------------
    # COO linear combination D(v)
    # -----------------------------
    def coo_linear_combo(self, v, drop_tol=0.0):
        """
        Build COO for D(v)=sum_l v_l D_l.

        Returns (rows, cols, vals) with duplicates summed.
        This is used inside Hv in Newton–CG.

        Uses precomputed concatenated arrays for O(nnz) vectorized operations.
        """
        if self._coo is None:
            raise ValueError("coo_linear_combo requires COO basis.")
        v = np.asarray(v, dtype=float).ravel()
        if v.size != self.m:
            raise ValueError("v has wrong length.")

        # Scale all vals by v[basis_idx] - fully vectorized
        scaled_vals = self._all_vals * v[self._basis_idx]

        # Use scipy.sparse to sum duplicates
        Dv = sparse.coo_matrix(
            (scaled_vals, (self._all_rows, self._all_cols)),
            shape=(self.n, self.n)
        )
        Dv.sum_duplicates()

        if drop_tol > 0:
            mask = np.abs(Dv.data) > drop_tol
            return Dv.row[mask], Dv.col[mask], Dv.data[mask]

        return Dv.row, Dv.col, Dv.data

    def sparse_linear_combo(self, v):
        """
        Build scipy.sparse CSR matrix for D(v)=sum_l v_l D_l.

        More efficient than coo_linear_combo when you need to do matrix-vector products.
        """
        if self._coo is None:
            raise ValueError("sparse_linear_combo requires COO basis.")
        v = np.asarray(v, dtype=float).ravel()
        if v.size != self.m:
            raise ValueError("v has wrong length.")

        scaled_vals = self._all_vals * v[self._basis_idx]
        Dv = sparse.csr_matrix(
            (scaled_vals, (self._all_rows, self._all_cols)),
            shape=(self.n, self.n)
        )
        return Dv


def solve_chol_multi_rhs(L, RHS):
    """
    Solve (L L^T) X = RHS for X, given Cholesky factor L.
    RHS can be (n,) or (n,r).
    """
    Y = np.linalg.solve(L, RHS)
    X = np.linalg.solve(L.T, Y)
    return X


# =============================================================================
# Banded Cholesky utilities (O(n·b²) instead of O(n³))
# =============================================================================

def detect_bandwidth(M, tol=1e-14):
    """
    Detect the bandwidth of a matrix M.
    Returns (lower_bw, upper_bw) where bandwidth = max(lower_bw, upper_bw).
    For symmetric matrices, lower_bw == upper_bw.
    """
    M = np.asarray(M)
    n = M.shape[0]
    lower_bw = 0
    upper_bw = 0
    for i in range(n):
        for j in range(i):
            if abs(M[i, j]) > tol:
                lower_bw = max(lower_bw, i - j)
        for j in range(i + 1, n):
            if abs(M[i, j]) > tol:
                upper_bw = max(upper_bw, j - i)
    return lower_bw, upper_bw


def dense_to_banded_lower(M, bandwidth):
    """
    Convert dense symmetric matrix M to lower-banded storage format for scipy.

    Returns ab of shape (bandwidth+1, n) where:
      ab[k, j] = M[j+k, j] for k = 0, ..., bandwidth

    This is the format expected by scipy.linalg.cholesky_banded with lower=True.
    """
    M = np.asarray(M)
    n = M.shape[0]
    b = bandwidth
    ab = np.zeros((b + 1, n), dtype=float)
    for k in range(b + 1):
        for j in range(n - k):
            ab[k, j] = M[j + k, j]
    return ab


def banded_to_dense_lower(ab, n):
    """
    Convert lower-banded storage back to dense symmetric matrix.
    """
    b = ab.shape[0] - 1
    M = np.zeros((n, n), dtype=float)
    for k in range(b + 1):
        for j in range(n - k):
            M[j + k, j] = ab[k, j]
            M[j, j + k] = ab[k, j]  # symmetric
    return M


def cholesky_banded(M, bandwidth):
    """
    Compute Cholesky factorization of banded SPD matrix M.

    Returns (Lb, bandwidth) where Lb is in lower-banded storage format.
    Complexity: O(n · bandwidth²) instead of O(n³).
    """
    ab = dense_to_banded_lower(M, bandwidth)
    cb = sp_linalg.cholesky_banded(ab, lower=True)
    return cb, bandwidth


def solve_chol_banded_multi_rhs(Lb, bandwidth, RHS):
    """
    Solve (L L^T) X = RHS using banded Cholesky factor Lb.

    Lb is in lower-banded storage format from cholesky_banded.
    Complexity: O(n · bandwidth) per RHS column.
    """
    RHS = np.asarray(RHS)
    if RHS.ndim == 1:
        return sp_linalg.cho_solve_banded((Lb, True), RHS)
    else:
        # Multi-RHS: solve column by column
        n, r = RHS.shape
        X = np.empty((n, r), dtype=float)
        for j in range(r):
            X[:, j] = sp_linalg.cho_solve_banded((Lb, True), RHS[:, j])
        return X


class CholeskyBackend:
    """
    Abstraction for Cholesky factorization supporting different matrix structures.

    Supported structures:
      - "dense": standard O(n³) Cholesky (default)
      - "banded": O(n·b²) banded Cholesky, requires bandwidth parameter
      - "tridiagonal": special case of banded with bandwidth=1

    Usage:
        backend = CholeskyBackend("banded", bandwidth=3)
        L = backend.factor(M)
        X = backend.solve(L, RHS)
    """

    def __init__(self, structure="dense", bandwidth=None):
        """
        Parameters
        ----------
        structure : str
            "dense", "banded", or "tridiagonal"
        bandwidth : int, optional
            Required for "banded". For "tridiagonal", bandwidth=1 is used.
        """
        self.structure = structure
        if structure == "tridiagonal":
            self.bandwidth = 1
        elif structure == "banded":
            if bandwidth is None:
                raise ValueError("bandwidth required for banded structure")
            self.bandwidth = bandwidth
        else:
            self.bandwidth = None

    def factor(self, M):
        """
        Compute Cholesky factorization of M.

        Returns a factor object that can be passed to solve().
        """
        if self.structure == "dense":
            L = np.linalg.cholesky(M)
            return ("dense", L)
        elif self.structure in ("banded", "tridiagonal"):
            Lb, bw = cholesky_banded(M, self.bandwidth)
            return ("banded", Lb, bw)
        else:
            raise ValueError(f"Unknown structure: {self.structure}")

    def solve(self, factor, RHS):
        """
        Solve (L L^T) X = RHS given factor from factor().
        """
        if factor[0] == "dense":
            _, L = factor
            return solve_chol_multi_rhs(L, RHS)
        elif factor[0] == "banded":
            _, Lb, bw = factor
            return solve_chol_banded_multi_rhs(Lb, bw, RHS)
        else:
            raise ValueError(f"Unknown factor type: {factor[0]}")

    def logdet(self, factor):
        """
        Compute log-determinant from Cholesky factor.
        log|M| = 2 * sum(log(diag(L)))
        """
        if factor[0] == "dense":
            _, L = factor
            return 2.0 * np.sum(np.log(np.diag(L)))
        elif factor[0] == "banded":
            _, Lb, bw = factor
            # First row of Lb contains diagonal of L
            return 2.0 * np.sum(np.log(Lb[0, :]))
        else:
            raise ValueError(f"Unknown factor type: {factor[0]}")


# =============================================================================
# Matrix structure detection
# =============================================================================

def is_toeplitz(A, tol=1e-10):
    """
    Check if matrix A is Toeplitz (constant along diagonals).

    Toeplitz means A[i,j] depends only on (i-j).
    """
    A = np.asarray(A)
    n = A.shape[0]
    for d in range(-n + 1, n):
        if d >= 0:
            diag_vals = [A[i, i + d] for i in range(n - d)]
        else:
            diag_vals = [A[i - d, i] for i in range(n + d)]
        if len(diag_vals) > 1:
            if np.max(np.abs(np.array(diag_vals) - diag_vals[0])) > tol * (1 + np.abs(diag_vals[0])):
                return False
    return True


def is_circulant(A, tol=1e-10):
    """
    Check if matrix A is circulant (each row is cyclic shift of previous).

    Circulant is a special case of Toeplitz where A[i,j] = c[(j-i) mod n].
    """
    A = np.asarray(A)
    n = A.shape[0]
    if n < 2:
        return True
    c = A[0, :]  # first row
    for i in range(1, n):
        expected = np.roll(c, i)
        if not np.allclose(A[i, :], expected, atol=tol, rtol=0):
            return False
    return True


def detect_matrix_structure(A, tol=1e-10):
    """
    Detect the structure of matrix A.

    Returns a dict with:
      - 'structure': one of 'dense', 'banded', 'tridiagonal', 'diagonal', 'toeplitz', 'circulant'
      - 'bandwidth': int (for banded/tridiagonal)
      - 'sparsity': fraction of zeros

    Priority order (most specific first):
      diagonal < tridiagonal < banded < toeplitz < circulant < dense
    """
    A = np.asarray(A)
    n = A.shape[0]

    # Compute bandwidth
    bw_lower, bw_upper = detect_bandwidth(A, tol=tol)
    bw = max(bw_lower, bw_upper)

    # Count non-zeros
    nnz = np.sum(np.abs(A) > tol)
    sparsity = 1.0 - nnz / (n * n)

    result = {
        'bandwidth': bw,
        'sparsity': sparsity,
        'n': n,
    }

    # Check from most specific to least
    if bw == 0:
        result['structure'] = 'diagonal'
    elif bw == 1:
        result['structure'] = 'tridiagonal'
    elif bw < n // 4:  # heuristic: banded if bandwidth < n/4
        result['structure'] = 'banded'
    elif is_circulant(A, tol=tol):
        result['structure'] = 'circulant'
    elif is_toeplitz(A, tol=tol):
        result['structure'] = 'toeplitz'
    else:
        result['structure'] = 'dense'

    return result


def auto_select_cholesky_backend(A, basis, verbose=False):
    """
    Automatically select the best Cholesky backend given A and basis.

    Checks:
      1. Structure of A (banded, Toeplitz, etc.)
      2. Whether basis matrices preserve the structure
      3. Cost-benefit of specialized vs dense Cholesky

    Returns:
      CholeskyBackend or None (None means use default dense)

    Usage:
      backend = auto_select_cholesky_backend(A, basis, verbose=True)
      B, C, x = constrained_decomposition(A, basis, cholesky_backend=backend)
    """
    A = np.asarray(A)
    n = A.shape[0]

    # Detect A's structure
    A_info = detect_matrix_structure(A)
    A_struct = A_info['structure']
    A_bw = A_info['bandwidth']

    # Get basis bandwidth
    basis_bw = basis.max_bandwidth()

    if verbose:
        print(f"[Auto-detect] A: structure={A_struct}, bandwidth={A_bw}, n={n}")
        print(f"[Auto-detect] Basis: m={basis.m}, max_bandwidth={basis_bw}")

    # Check if banded optimization applies
    # M = A - C(x) is banded iff both A and all Dk have bandwidth <= b
    if A_struct in ('diagonal', 'tridiagonal', 'banded') and basis_bw <= A_bw:
        effective_bw = max(A_bw, basis_bw)

        # Cost-benefit: banded is O(n * b²), dense is O(n³)
        # Banded wins when b² << n²
        banded_cost = n * effective_bw ** 2
        dense_cost = n ** 3 / 3  # approximate

        if banded_cost < dense_cost * 0.5:  # at least 2x speedup
            if verbose:
                speedup = dense_cost / banded_cost
                print(f"[Auto-detect] Using BANDED Cholesky (bandwidth={effective_bw}, ~{speedup:.0f}x speedup)")
            if effective_bw == 1:
                return CholeskyBackend("tridiagonal")
            else:
                return CholeskyBackend("banded", bandwidth=effective_bw)

    # TODO: Add Toeplitz support when scipy.linalg.solve_toeplitz is robust enough
    # For now, Toeplitz falls through to dense

    if verbose:
        print(f"[Auto-detect] Using DENSE Cholesky (no structure benefit detected)")

    return None  # Use default dense


def cg_solve_simple(matvec, b, tol=1e-6, max_iter=200):
    """
    Basic CG for SPD system A x = b, given matvec(x)=A@x.
    Stops when ||r|| <= tol*||b||. (Simpler version without preconditioner.)
    """
    b = np.asarray(b, dtype=float).ravel()
    m = b.size
    x = np.zeros(m, dtype=float)

    r = b - matvec(x)
    p = r.copy()
    rr = float(r @ r)
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return x, {"iters": 0, "converged": True, "rel_res": 0.0}

    thresh = (tol * b_norm) ** 2

    for it in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if denom <= 0:
            break
        alpha = rr / denom
        x += alpha * p
        r -= alpha * Ap
        rr_new = float(r @ r)
        if rr_new <= thresh:
            return x, {"iters": it, "converged": True, "rel_res": float(np.sqrt(rr_new) / b_norm)}
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return x, {"iters": it, "converged": False, "rel_res": float(np.sqrt(rr) / b_norm)}


def phi_grad_spd_implicit_selected(A, x, basis: SymBasis, cholesky_backend=None):
    """
    Compute phi and gradient g WITHOUT forming B.

    For COO basis:
      - factorize M = A - C(x) via Cholesky
      - compute only needed columns of B = M^{-1}:
          Z = B[:, J] where J is union of columns appearing in any Dk
      - compute g from Z using sparse trace accumulation

    For dense basis:
      - falls back to explicit B (no win possible without structure)

    Parameters
    ----------
    cholesky_backend : CholeskyBackend, optional
        If provided, uses specialized Cholesky (e.g., banded).
    """
    A = np.asarray(A, dtype=float)
    C = basis.build_C(x)
    M = A - C

    # Cholesky factorization (dense or specialized)
    if cholesky_backend is not None:
        factor = cholesky_backend.factor(M)
        phi = -cholesky_backend.logdet(factor)
    else:
        L = np.linalg.cholesky(M)
        phi = -_logdet_spd_from_cholesky(L)
        factor = ("dense", L)

    if not basis.is_sparse_coo():
        # dense fallback (no generic way to avoid B if you need all traces)
        n = A.shape[0]
        I = np.eye(n)
        if cholesky_backend is not None:
            B = cholesky_backend.solve(factor, I)
        else:
            B = solve_chol_multi_rhs(factor[1], I)
        B = 0.5 * (B + B.T)
        g = basis.trace_with(B)
        return phi, g, factor, C, M

    n = A.shape[0]
    J = basis.required_columns_for_traces()
    if J.size == 0:
        g = np.zeros(basis.m, dtype=float)
        return phi, g, factor, C, M

    E = np.eye(n)[:, J]           # n×|J|
    if cholesky_backend is not None:
        Z = cholesky_backend.solve(factor, E)
    else:
        Z = solve_chol_multi_rhs(factor[1], E)
    g = basis.trace_from_selected_cols(Z)
    return phi, g, factor, C, M


def hessvec_spd_coo_implicit_factory(factor, basis: SymBasis, drop_tol=0.0, cholesky_backend=None):
    """
    Build a closure Hv(v) that computes (H v) for the logdet objective,
    WITHOUT forming B and WITHOUT forming the full Hessian.

    COO-only version. Uses:
      I = union of all rows/cols that appear in any Dk
      Z = B[:, I] via multi-RHS solves (once per outer iteration)
      For each v:
         Dv = sum v_l D_l in COO
         Y = Dv @ Z  (sparse-dense)
         G = Z^T @ Y = Z^T Dv Z   (small r×r)
         Hv = basis.trace_from_small_G(G)

    Parameters
    ----------
    factor : tuple or ndarray
        Either a factor tuple from CholeskyBackend.factor(), or a dense L matrix
        (for backward compatibility).
    cholesky_backend : CholeskyBackend, optional
        If provided and factor is a tuple, uses backend.solve().
    """
    if not basis.is_sparse_coo():
        raise ValueError("Implicit COO Hessvec requires COO basis.")

    n = basis.n
    I = basis.required_indices_rows_cols()
    r = int(I.size)
    if r == 0:
        def Hv(v):
            return np.zeros(basis.m, dtype=float)
        return Hv

    E = np.eye(n)[:, I]           # n×r

    # Solve for Z = B[:, I]
    if cholesky_backend is not None and isinstance(factor, tuple):
        Z = cholesky_backend.solve(factor, E)
    elif isinstance(factor, tuple) and factor[0] == "dense":
        Z = solve_chol_multi_rhs(factor[1], E)
    else:
        # Backward compatibility: factor is just L
        Z = solve_chol_multi_rhs(factor, E)

    # Precompute mapping for trace extraction (use array version for vectorized lookup)
    I_pos_arr = basis._cached_I_pos_arr
    if I_pos_arr is None:
        basis.required_indices_rows_cols(rebuild_cache=True)
        I_pos_arr = basis._cached_I_pos_arr

    def Hv(v):
        v = np.asarray(v, dtype=float).ravel()
        # Build sparse D(v) and compute Y = Dv @ Z using scipy.sparse
        Dv = basis.sparse_linear_combo(v)
        if Dv.nnz == 0:
            return np.zeros(basis.m, dtype=float)

        # Y = Dv @ Z (n×r) - scipy.sparse handles this efficiently
        Y = Dv @ Z

        # G = Z^T @ Y (r×r)
        G = Z.T @ Y

        # Hv_k = sum_{(i,j) in Dk} Dk_ij * (B D(v) B)_{ij}
        # and (B D(v) B) restricted to I×I equals G
        return basis.trace_from_small_G(G, I_pos_arr=I_pos_arr)

    return Hv


def is_circulant(A, tol=1e-10):
    """
    Check if A is (approximately) circulant: each row is a cyclic shift of the first row.
    This is O(n^2); for large n, prefer to set `assume_circulant=True` in the circulant solver.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    r0 = A[0]
    for i in range(1, n):
        if not np.allclose(A[i], np.roll(r0, i), atol=tol, rtol=0):
            return False
    return True

def circulant_from_first_col(c):
    """
    Build the dense circulant matrix with first column c (length n).
    Column j is c shifted down by j.
    """
    c = np.asarray(c, dtype=float).ravel()
    n = c.size
    M = np.empty((n, n), dtype=float)
    for j in range(n):
        M[:, j] = np.roll(c, j)
    return M

def circulant_eigs_from_first_col(c):
    """
    Eigenvalues of circulant matrix from its first column, using numpy FFT convention.
    For real symmetric circulant matrices, these eigenvalues are real (up to roundoff).
    """
    c = np.asarray(c, dtype=float).ravel()
    return np.fft.fft(c)

def circulant_first_col_from_eigs(lam):
    """
    First column of circulant matrix from its eigenvalues (inverse FFT).
    """
    lam = np.asarray(lam)
    c = np.fft.ifft(lam)
    return np.real(c)

class CirculantSymBasis(SymBasis):
    """
    A SymBasis representing a subspace S of *circulant symmetric* matrices.

    Provide the basis via the first columns of D_k (length n each).
    We precompute FFT eigenvalues of each D_k to enable:
      - logdet/grad in O(n log n + m n)
      - full Hessian in O(n log n + m n + m^2 n)

    build_C(x) materializes a dense matrix only when needed for output/plotting.
    """
    def __init__(self, n, first_cols, dense_materialize=True, name="circulant"):
        self.n = int(n)
        self.m = len(first_cols)
        self.name = name
        self._dense_materialize = bool(dense_materialize)

        self._d_first_cols = [np.asarray(v, dtype=float).ravel() for v in first_cols]
        for v in self._d_first_cols:
            if v.size != self.n:
                raise ValueError("Each first_col must have length n.")

        lam = np.stack([circulant_eigs_from_first_col(v) for v in self._d_first_cols], axis=0)  # (m,n)
        if np.max(np.abs(np.imag(lam))) < 1e-10:
            lam = np.real(lam)
        self._lam_D = lam

        if self._dense_materialize:
            dense = []
            for v in self._d_first_cols:
                D = circulant_from_first_col(v)
                D = 0.5 * (D + D.T)
                dense.append(D)
            super().__init__(n=self.n, dense_mats=dense)
        else:
            super().__init__(n=self.n, dense_mats=[np.zeros((self.n, self.n))])

        self.is_circulant = True

    @property
    def lam_D(self):
        return self._lam_D

    @property
    def d_first_cols(self):
        return self._d_first_cols

    def build_C_first_col(self, x):
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.m:
            raise ValueError("x has wrong length for this basis.")
        c = np.zeros(self.n, dtype=float)
        for k in range(self.m):
            c += x[k] * self._d_first_cols[k]
        return c

    def build_C(self, x):
        c = self.build_C_first_col(x)
        return circulant_from_first_col(c)

def phi_grad_hess_spd_circulant(A, x, basis: CirculantSymBasis, order=1,
                               spd_eig_tol=1e-12, return_dense=True):
    """
    Fast evaluation for the circulant case via FFT diagonalization.
    """
    A = np.asarray(A, dtype=float)
    a0 = A[:, 0].copy()

    c0 = basis.build_C_first_col(x)
    m0 = a0 - c0

    lam_M = circulant_eigs_from_first_col(m0)
    if np.max(np.abs(np.imag(lam_M))) < 1e-10:
        lam_M = np.real(lam_M)

    if np.any(lam_M <= spd_eig_tol):
        raise np.linalg.LinAlgError("M(x) is not SPD (circulant eigenvalue check failed).")

    inv_lam = 1.0 / lam_M
    phi = -float(np.sum(np.log(lam_M)))

    lam_D = basis.lam_D
    if np.iscomplexobj(lam_D):
        g = np.real(lam_D @ inv_lam)
    else:
        g = lam_D @ inv_lam

    if order == 1:
        H = None
    else:
        w = inv_lam**2
        if np.iscomplexobj(lam_D):
            Vw = lam_D * w[None, :]
            H = np.real(Vw @ np.conjugate(lam_D).T)
        else:
            Vw = lam_D * w[None, :]
            H = Vw @ lam_D.T
        H = 0.5 * (H + H.T)

    if not return_dense:
        return phi, g, H, None, None, None, None

    C = circulant_from_first_col(c0)
    M = circulant_from_first_col(m0)
    b0 = circulant_first_col_from_eigs(inv_lam)
    B = circulant_from_first_col(b0)
    B = 0.5 * (B + B.T)

    return phi, g, H, B, C, M, None

class TridiagC_Basis(SymBasis):
    """
    Your specialized case:
      C_{k,k} = x_k
      C_{k,k+1} = C_{k+1,k} = -x_k/2
      C_{n-1,n-1} = 0
    with parameters x in R^{n-1} (m = n-1).

    In this case:
      g_k = tr(B Dk) = B_{k,k} - B_{k,k+1}
    and a fast explicit Hessian is available (your previous code).
    """

    def __init__(self, n):
        if n < 2:
            raise ValueError("n must be >= 2.")
        self.n = int(n)
        self.m = self.n - 1
        self.name = "tridiag_C_special"
        self._dense = None
        self._coo = None

    def build_C(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.m,):
            raise ValueError(f"x must have shape ({self.m},)")
        n = self.n
        C = np.zeros((n, n), dtype=float)
        for k in range(n - 1):
            C[k, k] += x[k]
            C[k, k + 1] += -0.5 * x[k]
            C[k + 1, k] += -0.5 * x[k]
        return C

    def trace_with(self, B):
        B = np.asarray(B, dtype=float)
        n = self.n
        g = np.empty(n - 1, dtype=float)
        for k in range(n - 1):
            g[k] = B[k, k] - B[k, k + 1]
        return g

    def hessian_from_B(self, B):
        return hessian_phi_spd_from_B(B)

def hessian_phi_spd_from_B(B):
    """
    Fast Hessian for the TridiagC_Basis case (ported from your code).
    """
    B = np.asarray(B, dtype=float)
    n = B.shape[0]
    m = n - 1
    H = np.zeros((m, m), dtype=float)

    # Diagonal
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

    # Off-diagonal
    for k in range(m):
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

        for l in range(k + 2, m):
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

def phi_grad_hess_spd(A, x, basis: SymBasis, order=1):
    """
    For the convex SPD formulation in your note:
        C(x) = sum x_k Dk
        M(x) = A - C(x)   must be SPD
        Phi(x) = -log det(M(x))
        B(x) = M(x)^{-1}
        grad_k = tr(B Dk)
        Hess_{k,l} = tr(B Dk B Dl)
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    
    # Fast path: circulant SPD + circulant basis -> FFT diagonalization
    if getattr(basis, "is_circulant", False):
        return phi_grad_hess_spd_circulant(A, x, basis, order=order, return_dense=True)

    C = basis.build_C(x)
    M = A - C

    # SPD check via Cholesky
    L = np.linalg.cholesky(M)
    phi = -_logdet_spd_from_cholesky(L)

    # Compute B = M^{-1} via solves (more stable than np.linalg.inv)
    I = np.eye(n)
    Y = np.linalg.solve(L, I)
    B = np.linalg.solve(L.T, Y)
    B = 0.5 * (B + B.T)

    g = basis.trace_with(B)

    if order == 1:
        return phi, g, None, B, C, M, L

    H = basis.hessian_from_B(B)
    if H is None:
        H = basis.generic_hessian_from_B(B)

    return phi, g, H, B, C, M, L


def phi_only_spd(A, x, basis: SymBasis):
    """
    Compute phi(x) = -log det(A - C(x)) with SPD feasibility check,
    but do NOT form B, gradient, or Hessian. Fast for line search.
    """
    A = np.asarray(A, dtype=float)
    C = basis.build_C(x)
    M = A - C
    L = np.linalg.cholesky(M)               # raises LinAlgError if infeasible
    phi = -_logdet_spd_from_cholesky(L)
    return phi



def cg_solve(matvec, b, x0=None, tol=1e-6, max_iter=200, M_inv=None):
    """
    Conjugate Gradient for SPD linear system A x = b, given only matvec(x)=A@x.

    Args:
      matvec: callable(v)->A v
      b: (m,) RHS
      x0: initial guess
      tol: relative tolerance on residual norm ||r|| <= tol*||b||
      max_iter: max CG iterations
      M_inv: optional preconditioner callable(z)->approx A^{-1} z

    Returns:
      x, info dict with keys: iters, converged, rel_res, abs_res
    """
    b = np.asarray(b, dtype=float).ravel()
    m = b.size
    if x0 is None:
        x = np.zeros(m, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()

    r = b - matvec(x)
    z = M_inv(r) if M_inv is not None else r.copy()
    p = z.copy()
    rz_old = float(r @ z)

    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return x, {"iters": 0, "converged": True, "rel_res": 0.0, "abs_res": 0.0}

    abs_tol = tol * b_norm

    for it in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if denom <= 0:
            # Should not happen for SPD, but damping/roundoff can cause this.
            break

        alpha = rz_old / denom
        x += alpha * p
        r -= alpha * Ap

        abs_res = float(np.linalg.norm(r))
        if abs_res <= abs_tol:
            return x, {"iters": it, "converged": True, "rel_res": abs_res / b_norm, "abs_res": abs_res}

        z = M_inv(r) if M_inv is not None else r
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    abs_res = float(np.linalg.norm(r))
    return x, {"iters": it, "converged": False, "rel_res": abs_res / b_norm, "abs_res": abs_res}


def hessvec_spd_from_B(v, B, basis: SymBasis):
    """
    Compute Hessian-vector product (Hv) for Phi(x)=-logdet(A-C(x)),
    using an explicit B = (A-C(x))^{-1}, but WITHOUT forming H.

    Identity:
      (Hv)_k = tr(B D_k B D(v)) = tr(D_k * (B D(v) B))
    where D(v) = sum_l v_l D_l.

    Implementation:
      Dv = basis.build_C(v)  (same linear combination)
      S  = B @ Dv @ B
      Hv = basis.trace_with(S)  # returns [ tr(S D_k) ]_k = [ tr(D_k S) ]_k

    Works for any basis that implements build_C and trace_with.
    """
    v = np.asarray(v, dtype=float).ravel()
    Dv = basis.build_C(v)          # dense matrix
    S = B @ Dv @ B                 # dense matrix
    Hv = basis.trace_with(S)       # length m
    return Hv


def constrained_decomposition(
    A,
    basis: SymBasis,
    tol=1e-8,
    max_iter=500,
    initial_step=1.0,
    backtracking_factor=0.5,
    armijo_alpha=1e-4,
    method="gradient-descent",
    verbose=False,
    newton_damping=1e-10,
    max_backtracks=60,
    max_m_for_full_hessian=500,
    log_prefix="",
    return_info=False,

    # Newton–CG controls
    cg_tol=1e-6,
    cg_max_iter=200,
    auto_newton_cg=True,

    # implicit Hv tuning
    hv_drop_tol=0.0,

    # Matrix structure optimization
    cholesky_backend=None,
    auto_backend=False,
):
    """
    Methods:
      - "gradient-descent"
      - "quasi-newton"
      - "newton"      (explicit Hessian; only practical for small m)
      - "newton-cg"   (matrix-free Newton with CG using implicit-B Hv)

    Auto-switch:
      If method=="newton" and m > max_m_for_full_hessian and auto_newton_cg=True,
      switch to "newton-cg".

    Implicit-B Newton–CG:
      For COO bases, Hv(v) is computed without forming B and without forming H.
      Uses only:
        - Cholesky of M
        - multi-RHS solves for selected columns of B
        - sparse COO algebra

    Matrix structure optimization:
      cholesky_backend : CholeskyBackend or None
        If provided, uses specialized Cholesky for structured matrices.
        Example for banded matrices:
          backend = CholeskyBackend("banded", bandwidth=3)
        For tridiagonal:
          backend = CholeskyBackend("tridiagonal")

      auto_backend : bool
        If True, automatically detect matrix structure and select the best
        Cholesky backend. Analyzes A and basis to determine if banded,
        tridiagonal, or other specialized factorizations can be used.
        Overrides cholesky_backend if set.

        Structure preservation: If A has bandwidth b and all basis matrices Dk
        have bandwidth <= b, then M = A - C(x) has bandwidth b, so banded
        Cholesky reduces cost from O(n³) to O(n·b²).
    """
    pfx = log_prefix or ""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square.")
    if not is_spd(A):
        raise ValueError("A must be symmetric positive definite for the SPD formulation.")
    if basis.n != n:
        raise ValueError(f"basis.n={basis.n} must match A.shape[0]={n}.")

    m = basis.m
    if m == 0:
        C = np.zeros_like(A)
        B = spd_inverse(A)
        x = np.zeros((0,), dtype=float)
        if return_info:
            return B, C, x, {"iters": 0, "backtracks": 0, "converged": True, "final_max_abs_trace": 0.0, "used_method": method}
        return B, C, x

    if method not in ("gradient-descent", "quasi-newton", "newton", "newton-cg"):
        raise ValueError("method must be one of {'gradient-descent','quasi-newton','newton','newton-cg'}.")

    # Auto-detect matrix structure and select best Cholesky backend
    if auto_backend:
        cholesky_backend = auto_select_cholesky_backend(A, basis, verbose=verbose)

    used_method = method
    if method == "newton" and auto_newton_cg and (m > max_m_for_full_hessian):
        used_method = "newton-cg"
        # Always print auto-switch message (important for user awareness)
        print(f"{pfx}[Auto-switch] m={m} > {max_m_for_full_hessian} -> using newton-cg (matrix-free)")

    x = np.zeros(m, dtype=float)

    # Initial evaluation (phi + g) without explicit B when possible
    if used_method in ("gradient-descent", "quasi-newton", "newton-cg"):
        phi, g, factor, C, M = phi_grad_spd_implicit_selected(A, x, basis, cholesky_backend=cholesky_backend)
        H = None
        g_prev = g.copy()  # for BFGS
    else:
        phi, g, H, B, C, M, L = phi_grad_hess_spd(A, x, basis, order=2)
        factor = ("dense", L) if L is not None else None
        g_prev = None

    if verbose and isinstance(factor, tuple) and factor[0] == "dense":
        print(f"{pfx}...  min(diag(L))={float(np.min(np.diag(factor[1]))):.3e}")

    if used_method == "quasi-newton":
        H_BFGS = np.eye(m)

    total_backtracks = 0
    iters_done = 0

    start_iter_time = time.perf_counter()
    for it in range(max_iter):
        iters_done = it + 1
        g_norm = float(np.linalg.norm(g))
        max_abs_g = float(np.max(np.abs(g)))

        if verbose:
            dt = time.perf_counter() - start_iter_time
            print(f"{pfx}iter {it:4d}  phi={phi: .6e}  ||g||={g_norm: .3e}  max|g|={max_abs_g: .3e} time={dt: .3e}")
            start_iter_time = time.perf_counter()

        if max_abs_g < tol:
            break

        # -------------------------
        # direction
        # -------------------------
        if used_method == "gradient-descent":
            d = -g

        elif used_method == "quasi-newton":
            d = -(H_BFGS @ g)

        elif used_method == "newton":
            if H is None:
                raise ValueError("Newton requested but Hessian is None.")
            lam = newton_damping
            for _ in range(10):
                try:
                    H_damped = H + lam * np.eye(m)
                    Lh = np.linalg.cholesky(H_damped)
                    y = np.linalg.solve(Lh, -g)
                    d = np.linalg.solve(Lh.T, y)
                    break
                except np.linalg.LinAlgError:
                    lam = max(10.0 * lam, 1e-12)
            else:
                d = -g

        else:
            # -------------------------
            # newton-cg with implicit-B Hv
            # -------------------------
            lam = max(newton_damping, 0.0)

            if basis.is_sparse_coo():
                Hv = hessvec_spd_coo_implicit_factory(factor, basis, drop_tol=hv_drop_tol, cholesky_backend=cholesky_backend)
            else:
                # dense basis fallback: Hv requires explicit B (no generic win)
                Ieye = np.eye(n)
                if cholesky_backend is not None:
                    B = cholesky_backend.solve(factor, Ieye)
                else:
                    B = solve_chol_multi_rhs(factor[1], Ieye)
                B = 0.5 * (B + B.T)
                def Hv(v):
                    # Hv = tr(Dk * (B D(v) B)) = trace_with(B D(v) B)
                    Dv = basis.build_C(v)
                    S = B @ Dv @ B
                    return basis.trace_with(S)

            def mv(v):
                return Hv(v) + lam * np.asarray(v, dtype=float)

            d, cg_info = cg_solve(mv, -g, tol=cg_tol, max_iter=min(cg_max_iter, max(10, m)))

            if verbose:
                print(f"{pfx}  [CG] iters={cg_info['iters']} converged={cg_info['converged']} rel_res={cg_info['rel_res']:.2e}")

            if (not cg_info["converged"]) and (cg_info["rel_res"] > 1e-2):
                d = -g

        # Ensure descent
        gTd = float(g @ d)
        if gTd >= 0:
            d = -g
            gTd = float(g @ d)

        # -------------------------
        # backtracking
        # -------------------------
        t = float(initial_step)
        accepted = False
        for _ in range(max_backtracks):
            total_backtracks += 1
            x_try = x + t * d
            try:
                phi_try = phi_only_spd(A, x_try, basis)
            except np.linalg.LinAlgError:
                t *= backtracking_factor
                continue
            if phi_try <= phi + armijo_alpha * t * gTd:
                accepted = True
                break
            t *= backtracking_factor

        if not accepted:
            break

        # -------------------------
        # accept + reevaluate (implicit)
        # -------------------------
        x_prev = x
        x = x_try
        phi, g, factor, C, M = phi_grad_spd_implicit_selected(A, x, basis, cholesky_backend=cholesky_backend)

        # BFGS update
        if used_method == "quasi-newton":
            s = x - x_prev
            y_vec = g - g_prev
            sy = float(s @ y_vec)
            if sy > 1e-12:
                rho = 1.0 / sy
                I_m = np.eye(m)
                V = I_m - rho * np.outer(s, y_vec)
                H_BFGS = V @ H_BFGS @ V.T + rho * np.outer(s, s)
            g_prev = g.copy()

    # finalize: return explicit B,C like before (compute B once at end)
    # (This is unavoidable if you insist on returning B explicitly.)
    Ieye = np.eye(n)
    if cholesky_backend is not None:
        B = cholesky_backend.solve(factor, Ieye)
    else:
        B = solve_chol_multi_rhs(factor[1], Ieye)
    B = 0.5 * (B + B.T)

    final_g = basis.trace_with(B)
    max_trace = float(np.max(np.abs(final_g))) if final_g.size else 0.0

    info = {
        "iters": iters_done,
        "backtracks": total_backtracks,
        "converged": (max_trace < tol),
        "final_max_abs_trace": max_trace,
        "used_method": used_method,
    }
    if return_info:
        return B, C, x, info
    return B, C, x


def make_coo_basis_from_sparse_patterns(n, patterns):
    """
    Helper for creating a SymBasis from patterns.

    patterns: list of list-of-triplets, where each basis matrix is described by
              [(i,j,val), ...] with 0-based indices.
    """
    coo = []
    for triplets in patterns:
        rows = [t[0] for t in triplets]
        cols = [t[1] for t in triplets]
        vals = [t[2] for t in triplets]
        coo.append((np.array(rows, dtype=int), np.array(cols, dtype=int), np.array(vals, dtype=float)))
    return SymBasis(n=n, coo_mats=coo)

def _sym_upper_indices(n: int):
    """Return arrays (rows, cols) for i<=j in row-major order."""
    rows, cols = np.triu_indices(n)
    return rows, cols

def _sym_vec(M: np.ndarray):
    """
    Vectorize symmetric matrix using upper triangle with sqrt(2) scaling off-diagonal
    so that <X,Y>_F = symvec(X)^T symvec(Y).
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    r, c = _sym_upper_indices(n)
    v = M[r, c].copy()
    off = (r != c)
    v[off] *= np.sqrt(2.0)
    return v

def _sym_unvec(v: np.ndarray, n: int):
    """Inverse of _sym_vec."""
    v = np.asarray(v, dtype=float)
    r, c = _sym_upper_indices(n)
    M = np.zeros((n, n), dtype=float)
    vv = v.copy()
    off = (r != c)
    vv[off] /= np.sqrt(2.0)
    M[r, c] = vv
    M[c, r] = vv
    return M

def _orthonormalize_dense_sym_basis(mats, atol=1e-12):
    """
    Orthonormalize a list of symmetric matrices in Frobenius inner product.
    Returns a (possibly shorter) list.
    """
    if len(mats) == 0:
        return []
    n = mats[0].shape[0]
    V = np.stack([_sym_vec(M) for M in mats], axis=1)  # (p, k)
    # QR with column pivoting via SVD (stable)
    U, s, VT = np.linalg.svd(V, full_matrices=False)
    keep = s > atol * s[0] if s.size else np.array([], dtype=bool)
    if not np.any(keep):
        return []
    # Orthonormal basis vectors in the column space:
    Q = U[:, keep]  # columns orthonormal
    return [_sym_unvec(Q[:, i], n) for i in range(Q.shape[1])]

def _phi_grad_hess_dual(A, y, basis_perp: SymBasis, order=1):
    r"""
    Dual objective over y:
        B(y) = sum_i y_i E_i   (E_i span S^\perp)
        psi(y) = -log det(B(y)) + tr(A B(y))
    Gradient:
        g_i = < -B^{-1} + A, E_i >
    Hessian:
        H_{ij} = tr(B^{-1} E_i B^{-1} E_j)
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    B = basis_perp.build_C(y)  # reuse build_C: linear combination
    B = 0.5 * (B + B.T)

    L = np.linalg.cholesky(B)  # feasibility
    psi = -_logdet_spd_from_cholesky(L) + float(np.sum(A * B))

    # Binv via solves
    I = np.eye(n)
    Y = np.linalg.solve(L, I)
    Binv = np.linalg.solve(L.T, Y)
    Binv = 0.5 * (Binv + Binv.T)

    # grad: <A - Binv, E_i>
    g = basis_perp.trace_with(A - Binv)

    if order == 1:
        return psi, g, None, B, Binv, L

    # Hessian: tr(B^{-1} E_i B^{-1} E_j)
    # This is the same form as primal Hessian with "B" replaced by Binv,
    # but here our linear maps are E_i.
    H = basis_perp.hessian_from_B(Binv)
    if H is None:
        H = basis_perp.generic_hessian_from_B(Binv)
    return psi, g, H, B, Binv, L

def _psi_only_dual(A, y, basis_perp: SymBasis):
    A = np.asarray(A, dtype=float)
    B = basis_perp.build_C(y)
    B = 0.5 * (B + B.T)
    L = np.linalg.cholesky(B)
    psi = -_logdet_spd_from_cholesky(L) + float(np.sum(A * B))
    return psi

def _find_feasible_dual_start(A, basis_perp: SymBasis, tries=30, jitter=1e-6, rng=None):
    r"""
    Heuristic to find y0 such that B(y0) is SPD.
    Prefers something 'close' to I and/or A^{-1}, projected to S^\perp.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    # candidate targets to project onto span(E)
    targets = [np.eye(n), spd_inverse(A)]
    # add a few random SPD-ish targets
    for _ in range(3):
        M = rng.standard_normal((n, n))
        targets.append(M @ M.T + n * np.eye(n))

    # Since basis_perp is orthonormal-ish only if constructed by make_orthogonal...,
    # we solve least squares y minimizing ||B(y)-T||_F.
    # Build matrix of symvec(Ei) columns.
    Emats = basis_perp._dense
    V = np.stack([_sym_vec(E) for E in Emats], axis=1)  # (p_total, p)
    for T in targets:
        t = _sym_vec(0.5 * (T + T.T))
        y_ls, *_ = np.linalg.lstsq(V, t, rcond=None)
        # try scaled versions + jitter along any SPD direction inside span(E)
        for scale in (1.0, 0.3, 3.0, 10.0):
            y0 = scale * y_ls
            B0 = basis_perp.build_C(y0)
            B0 = 0.5 * (B0 + B0.T)
            # add tiny diagonal component projected into span(E)
            # (use projection of I onto span(E) itself)
            Iproj = basis_perp.build_C(np.linalg.lstsq(V, _sym_vec(np.eye(n)), rcond=None)[0])
            Iproj = 0.5 * (Iproj + Iproj.T)
            for t_j in (0.0, jitter, 10*jitter, 100*jitter):
                try:
                    np.linalg.cholesky(B0 + t_j * Iproj)
                    return y0 + t_j * np.linalg.lstsq(V, _sym_vec(Iproj), rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass

    # final resort: random coefficients until SPD
    p = basis_perp.m
    for _ in range(tries):
        y0 = rng.standard_normal(p)
        B0 = basis_perp.build_C(y0)
        B0 = 0.5 * (B0 + B0.T)
        # try to make SPD by adding positive multiple of Iproj
        Iproj = basis_perp.build_C(np.linalg.lstsq(V, _sym_vec(np.eye(n)), rcond=None)[0])
        Iproj = 0.5 * (Iproj + Iproj.T)
        for t_j in (jitter, 10*jitter, 100*jitter, 1e-2, 1e-1, 1.0):
            try:
                np.linalg.cholesky(B0 + t_j * Iproj)
                # convert to y by least squares
                y_fix, *_ = np.linalg.lstsq(V, _sym_vec(B0 + t_j * Iproj), rcond=None)
                return y_fix
            except np.linalg.LinAlgError:
                pass

    raise RuntimeError("Could not find feasible starting point for dual (B(y) SPD).")

def make_orthogonal_complement_basis(basis: SymBasis, atol=1e-12, name=None):
    r"""
    Construct a dense SymBasis spanning S^\perp where S = span(D1,...,Dm)
    is given by `basis`. Works best for moderate n.

    Returns a SymBasis with dense_mats=[E1,...,Ep] such that tr(Ei Dk)=0 for all k.
    """
    n = basis.n
    # Build V_S = [symvec(D1) ... symvec(Dm)]
    Dmats = []
    if basis._dense is not None:
        Dmats = [np.asarray(D, dtype=float) for D in basis._dense]
    else:
        # COO -> dense
        for (rows, cols, vals) in basis._coo:
            D = np.zeros((n, n), dtype=float)
            D[rows, cols] += vals
            Dmats.append(D)

    V = np.stack([_sym_vec(D) for D in Dmats], axis=1)  # (p_total, m)

    # Nullspace of V^T (vectors orthogonal to all Dk)
    # Use SVD on V^T: V^T = U S W^T ; nullspace basis are columns of W with s ~ 0.
    U, s, WT = np.linalg.svd(V.T, full_matrices=True)
    # WT has shape (p_total, p_total). Singular values length = m.
    if s.size == 0:
        # S is empty => S={0} => S^\perp is all symmetric matrices
        W = WT.T
        null = W
    else:
        tol = atol * s[0]
        rank = int(np.sum(s > tol))
        W = WT.T
        null = W[:, rank:]  # (p_total, p_total-rank)

    Emats = [_sym_unvec(null[:, i], n) for i in range(null.shape[1])]
    Emats = _orthonormalize_dense_sym_basis(Emats, atol=atol)

    if name is None:
        name = f"{basis.name}_perp"
    return SymBasis(n=n, dense_mats=Emats, name=name)

def project_onto_subspace(M, basis: SymBasis, gram_inv=None):
    """
    Frobenius projection of symmetric matrix M onto S=span(Dk).
    Returns (proj, coeffs).
    """
    M = np.asarray(M, dtype=float)
    n = basis.n
    if M.shape != (n, n):
        raise ValueError("M shape mismatch.")
    # Build dense Dk list
    if basis._dense is None:
        Dmats = []
        for (rows, cols, vals) in basis._coo:
            D = np.zeros((n, n), dtype=float)
            D[rows, cols] += vals
            Dmats.append(D)
    else:
        Dmats = basis._dense

    m = basis.m
    b = np.array([np.sum(M * Dmats[k]) for k in range(m)], dtype=float)  # <M,Dk>
    if gram_inv is None:
        G = np.array([[np.sum(Dmats[i] * Dmats[j]) for j in range(m)] for i in range(m)], dtype=float)
        # Regularize if ill-conditioned
        G = 0.5 * (G + G.T)
        G += 1e-14 * np.eye(m)
        gram_inv = np.linalg.inv(G)
    coeffs = gram_inv @ b
    P = np.zeros_like(M)
    for k in range(m):
        P += coeffs[k] * Dmats[k]
    return P, coeffs

def constrained_decomposition_dual(
    A,
    basis: SymBasis,
    basis_perp: SymBasis = None,
    tol=1e-8,
    max_iter=300,
    initial_step=1.0,
    backtracking_factor=0.5,
    armijo_alpha=1e-4,
    verbose=False,
    newton_damping=1e-10,
    max_backtracks=60,
    atol_perp=1e-12,
    log_prefix="",
    return_info=False
):
    r"""
    Dual Newton solver (mirrors constrained_decomposition for the primal).

    Solves:
        minimize_B  -log det(B) + tr(A B)
        s.t.         B > 0,   tr(B Dk)=0  (i.e., B in S^\perp)

    Inputs:
      - A: SPD matrix
      - basis: SymBasis spanning S (optional if basis_perp is provided and S is huge)
      - basis_perp: optional SymBasis spanning S^\perp.
        If None, this implementation REQUIRES you to provide it explicitly
        (we do not auto-construct S^\perp from S).

    Output:
      (B, C, y, basis_perp) where:
        B > 0 and B perpendicular to S,
        C in S and A = B^{-1} + C
        y are the coordinates of B in basis_perp.
    """
    # Prefix for all verbose logging produced by this solver (e.g. per-demo tags).
    pfx = log_prefix or ""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if not is_spd(A):
        raise ValueError("A must be symmetric positive definite.")

    if basis is None and basis_perp is None:
        raise ValueError("Provide either `basis` (for S) or `basis_perp` (for S^perp).")

    if basis is not None and basis.n != n:
        raise ValueError(f"basis.n={basis.n} must match A.shape[0]={n}.")
    if basis_perp is not None and basis_perp.n != n:
        raise ValueError(f"basis_perp.n={basis_perp.n} must match A.shape[0]={n}.")

    # ------------------------------------------------------------------
    # ### FIX 1: Do NOT call a non-existent orthogonal_complement_basis.
    # If user didn't provide basis_perp, we cannot proceed (unless you
    # later implement make_orthogonal_complement_basis).
    # ------------------------------------------------------------------
    if basis_perp is None:
        raise ValueError(
            "basis_perp is None. Please provide an explicit basis for S^perp "
            "(e.g. banded/tridiag basis). Automatic construction from `basis` "
            "is not implemented."
        )

    # Find feasible start y0
    y = _find_feasible_dual_start(A, basis_perp)

    psi, g, H, B, Binv, L = _phi_grad_hess_dual(A, y, basis_perp, order=2)

    total_backtracks = 0
    iters_done = 0

    for it in range(max_iter):
        iters_done = it + 1
        g_norm = float(np.linalg.norm(g))
        max_abs_g = float(np.max(np.abs(g))) if g.size else 0.0
        if verbose:
            print(f"{pfx}[dual] iter {it:4d}  psi={psi: .6e}  ||g||={g_norm: .3e}  max|g|={max_abs_g: .3e}")

        if max_abs_g < tol:
            if verbose:
                print(f"{pfx}[dual] Converged: max|grad|={max_abs_g:.3e} < tol={tol}")
            break

        p = basis_perp.m

        # Newton direction: (H+λI)d = -g
        lam = newton_damping
        for _ in range(10):
            try:
                H_damped = H + lam * np.eye(p)
                Lh = np.linalg.cholesky(H_damped)
                ytmp = np.linalg.solve(Lh, -g)
                d = np.linalg.solve(Lh.T, ytmp)
                break
            except np.linalg.LinAlgError:
                lam = max(10.0 * lam, 1e-12)
        else:
            d = -g

        gTd = float(g @ d)
        if gTd >= 0:
            d = -g
            gTd = float(g @ d)

        # backtracking with feasibility and Armijo on psi
        t = float(initial_step)
        accepted = False
        for _ in range(max_backtracks):
            total_backtracks += 1
            y_try = y + t * d
            try:
                psi_try = _psi_only_dual(A, y_try, basis_perp)
            except np.linalg.LinAlgError:
                t *= backtracking_factor
                continue

            if psi_try <= psi + armijo_alpha * t * gTd:
                accepted = True
                break
            t *= backtracking_factor

        if not accepted:
            if verbose:
                print(f"{pfx}[dual] Backtracking failed; stopping.")
            break

        psi, g, H, B, Binv, L = _phi_grad_hess_dual(A, y_try, basis_perp, order=2)
        y = y_try

    # ------------------------------------------------------------------
    # ### FIX 2: Actually form C before trying to project it.
    # ------------------------------------------------------------------
    C = A - Binv  # since at optimum: A = B^{-1} + C

    # Optional: project C onto S to clean numerical noise.
    # For very large S, basis may be None or not materializable; then skip.
    if basis is not None and getattr(basis, "m", 0) <= 2000 and getattr(basis, "dense_mats", None) is not None:
        P_S, _ = project_onto_subspace(C, basis)
        C = 0.5 * (P_S + P_S.T)
    else:
        C = 0.5 * (C + C.T)

    final_max_abs_grad = float(np.max(np.abs(g))) if g.size else 0.0
    info = {
        "iters": iters_done,
        "backtracks": total_backtracks,
        "converged": (final_max_abs_grad < tol) if g.size else True,
        "final_max_abs_grad": final_max_abs_grad,
    }
    if return_info:
        return B, C, y, basis_perp, info
    return B, C, y, basis_perp

def _as_perm_matrix(perm, n):
    """perm can be (n,) int array representing a permutation, or an (n,n) matrix."""
    P = np.asarray(perm)
    if P.shape == (n,):
        perm = P.astype(int)
        if np.any(perm < 0) or np.any(perm >= n) or len(np.unique(perm)) != n:
            raise ValueError("Invalid permutation array.")
        M = np.zeros((n, n), dtype=float)
        M[np.arange(n), perm] = 1.0
        return M
    if P.shape == (n, n):
        return P.astype(float)
    raise ValueError("perm must be a permutation vector of length n or an (n,n) matrix.")

def group_average_conjugation(M, group, n=None):
    """
    Average of conjugations:  (1/|G|) sum_{P in G} P M P^T

    Supported `group` formats:
      1) {"blocks": blocks}  -> fast closed-form Reynolds projection onto the
         within-block permutation fixed space (uses block_reynolds_project).
         Here `blocks` is a list of index arrays/lists.
      2) list/iterable of permutation vectors or permutation matrices -> explicit averaging.
    """
    M = np.asarray(M, dtype=float)
    if n is None:
        n = M.shape[0]

    # Fast exact averaging for block-permutation group
    if isinstance(group, dict) and ("blocks" in group):
        out = block_reynolds_project(M, group["blocks"])
        return 0.5 * (out + out.T)

    # Explicit finite-group averaging (small groups only)
    out = np.zeros_like(M, dtype=float)
    cnt = 0
    for g in group:
        P = _as_perm_matrix(g, n)
        out += P @ M @ P.T
        cnt += 1
    if cnt == 0:
        raise ValueError("group_average_conjugation: empty group.")
    out /= float(cnt)
    return 0.5 * (out + out.T)


def make_group_invariant_basis(basis: SymBasis, group, atol=1e-12, name=None):
    """
    Build a new SymBasis spanning the G-invariant part of S:
        S^G = { X in S : P X P^T = X for all P in G }
    by averaging each basis matrix under the group and then orthonormalizing / pruning.
    """
    n = basis.n
    # Dense Dk list
    if basis._dense is None:
        Dmats = []
        for (rows, cols, vals) in basis._coo:
            D = np.zeros((n, n), dtype=float)
            D[rows, cols] += vals
            Dmats.append(D)
    else:
        Dmats = [np.asarray(D, dtype=float) for D in basis._dense]

    averaged = [group_average_conjugation(D, group, n=n) for D in Dmats]
    inv_mats = _orthonormalize_dense_sym_basis(averaged, atol=atol)

    if name is None:
        name = f"{basis.name}_Ginv"
    if len(inv_mats) == 0:
        raise ValueError("Group-invariant subspace is {0}. Nothing to optimize.")
    return SymBasis(n=n, dense_mats=inv_mats, name=name)


def constrained_decomposition_group_invariant(
    A,
    basis: SymBasis,
    group,
    solver="primal",
    method="newton",
    tol=1e-8,
    max_iter=500,
    verbose=False,
    return_info=False,
    log_prefix="",
    enforce_A_fixed=True,        # theorem mode
    project_A_if_needed=False,   # robustness mode (off by default)
    invariant_tol=1e-10,
    **kwargs,
):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    # Reduce to S^G
    basis_G = make_group_invariant_basis(basis, group)

    # Check invariance of A (do NOT change A by default)
    A_proj = group_average_conjugation(A, group, n=n)
    rel = np.linalg.norm(A - A_proj, ord="fro") / max(1.0, np.linalg.norm(A, ord="fro"))

    A_use = A
    if enforce_A_fixed and rel > invariant_tol:
        if project_A_if_needed:
            A_use = A_proj
            if verbose:
                print(f"{log_prefix}[group] A not G-fixed (rel={rel:.2e}); using Pi_G(A).")
        else:
            raise ValueError(
                f"A is not G-fixed: rel||A-Pi_G(A)||_F={rel:.2e} > {invariant_tol:.2e}. "
                "Either pass invariant A, or set project_A_if_needed=True."
            )

    if solver == "primal":
        out = constrained_decomposition(
            A=A_use,
            basis=basis_G,
            method=method,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            log_prefix=log_prefix,
            return_info=return_info,
            **kwargs,
        )
        if return_info:
            B, C, x, info = out
            info = dict(info)
            info["m_G"] = getattr(basis_G, "m", None)
            info["A_proj_rel"] = float(rel)
            info["used_A_projection"] = bool(A_use is A_proj)
            return B, C, x, basis_G, info
        else:
            B, C, x = out
            return B, C, x, basis_G

    if solver == "dual":
        out = constrained_decomposition_dual(
            A=A_use,
            basis=basis_G,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            log_prefix=log_prefix,
            return_info=return_info,
            **kwargs,
        )
        if return_info:
            B, C, y, basis_perp, info = out
            info = dict(info)
            info["m_G"] = getattr(basis_G, "m", None)
            info["A_proj_rel"] = float(rel)
            info["used_A_projection"] = bool(A_use is A_proj)
            return B, C, y, basis_G, basis_perp, info
        else:
            B, C, y, basis_perp = out
            return B, C, y, basis_G, basis_perp

    raise ValueError("solver must be 'primal' or 'dual'.")


def make_symmetric_first_col_from_half(half, n):
    half = np.asarray(half, dtype=float)  # length n//2+1 when n even
    if n % 2 == 0:
        # half = [a0, a1, ..., a_{n/2}]
        # full = [a0, a1, ..., a_{n/2}, a_{n/2-1}, ..., a1]
        full = np.concatenate([half, half[(n//2-1):0:-1]])
    else:
        # half = [a0, a1, ..., a_{(n-1)/2}]
        # full = [a0, a1, ..., a_{(n-1)/2}, a_{(n-1)/2}, ..., a1]
        full = np.concatenate([half, half[:0:-1]])
    return full

def shift_to_spd_full_first_col(first_col, eps=1e-9):
    a = np.asarray(first_col, dtype=float).copy()
    lam = np.fft.fft(a).real          # eigenvalues of the n×n circulant
    min_lam = lam.min()
    if min_lam <= eps:
        a[0] += (eps - min_lam)       # shift all eigenvalues up so min is eps
    return a

def constrained_decomposition_circulant(
    A,
    first_cols,
    *,
    method="newton",
    tol=1e-8,
    max_iter=50,
    verbose=False,
    **kwargs,
):
    """
    Returns (B, C, x, basis)
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    basis = CirculantSymBasis(n, first_cols, dense_materialize=True)

    B, C, x = constrained_decomposition(
        A,
        basis,
        method=method,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
        **kwargs,
    )
    return B, C, x, basis
