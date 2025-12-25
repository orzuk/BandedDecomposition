import argparse
import os
import time
import numpy as np
from constrained_decomposition_utils import *


# ============================================================
# Utilities
# ============================================================




def _logdet_spd_from_cholesky(L):
    return 2.0 * np.sum(np.log(np.diag(L)))


# ============================================================
# Basis representations for S = span(D1,...,Dm)
# ============================================================

class SymBasis:
    """
    Represents a symmetric matrix subspace S = span(D1,...,Dm).

    Goals:
      1) Be general (any symmetric basis matrices).
      2) Stay efficient when the basis is structured/sparse.

    You can provide the basis in one of three ways:

    (A) dense_mats: list of (n,n) dense numpy arrays.
    (B) coo_mats: list where each element is a tuple (rows, cols, vals, n),
        describing a sparse matrix in COO format. (rows, cols, vals are 1D arrays).
        NOTE: you may include both (i,j) and (j,i) entries if you like; we do not
        automatically symmetrize sparse inputs for speed.
    (C) custom builder / tracers:
        - build_C(x): override to build C(x) quickly without iterating full basis
        - trace_with(B): override to compute g_k = tr(B Dk) quickly.

    If you pass dense_mats or coo_mats, default implementations are used.
    """

    def __init__(self, n, dense_mats=None, coo_mats=None, name="generic"):
        self.n = int(n)
        self.name = name

        if dense_mats is None and coo_mats is None:
            raise ValueError("Provide either dense_mats or coo_mats (or subclass SymBasis).")

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
            # Each Dk described by (rows, cols, vals) with 0-based indices.
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

    # -----------------------------
    # Default linear map C(x)
    # -----------------------------
    def build_C(self, x):
        """
        Build C(x) = sum_k x_k Dk as a dense (n,n) array.
        Override if you can do better in your structured case.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.m,):
            raise ValueError(f"x must have shape ({self.m},)")
        C = np.zeros((self.n, self.n), dtype=float)

        if self._dense is not None:
            # Dense accumulation
            for k, Dk in enumerate(self._dense):
                C += x[k] * Dk
            return C

        # Sparse COO accumulation
        for k, (rows, cols, vals) in enumerate(self._coo):
            C[rows, cols] += x[k] * vals
        return C

    # -----------------------------
    # Default tracer g_k = tr(B Dk)
    # -----------------------------
    def trace_with(self, B):
        """
        Compute g_k = tr(B Dk) for k=1..m.
        For SPD B and symmetric Dk, this is also <B, Dk>_F.

        Override if you have a fast formula.
        """
        B = np.asarray(B, dtype=float)
        if B.shape != (self.n, self.n):
            raise ValueError("B has wrong shape.")
        g = np.zeros(self.m, dtype=float)

        if self._dense is not None:
            for k, Dk in enumerate(self._dense):
                g[k] = float(np.sum(B * Dk))
            return g

        # sparse COO: trace = sum_{i,j} B_ij Dk_ij
        for k, (rows, cols, vals) in enumerate(self._coo):
            g[k] = float(np.sum(B[rows, cols] * vals))
        return g

    # -----------------------------
    # Optional specialized Hessian
    # -----------------------------
    def hessian_from_B(self, B):
        """
        If you can compute the Hessian H_{k,l} = tr(B Dk B Dl) faster
        than generic O(m^2 n^2), override this method.

        By default, returns None which triggers the generic path.
        """
        return None

    # -----------------------------
    # Generic Hessian builder (fallback)
    # -----------------------------
    def generic_hessian_from_B(self, B, max_m_for_full=1000):
        """
        Generic Hessian H_{k,l} = tr(B Dk B Dl).

        This forms the full m x m Hessian and is intended for small/moderate m.
        """
        m = self.m
        n = self.n
        if m > max_m_for_full:
            raise ValueError(
                f"m={m} is large; forming the full Hessian is expensive. "
                f"Either use quasi-newton, reduce m, or implement hessian_from_B."
            )

        B = np.asarray(B, dtype=float)

        # Build BDk = B @ Dk efficiently.
        # Dense basis: standard multiplication.
        if self._dense is not None:
            BD = np.stack([B @ Dk for Dk in self._dense], axis=0)  # (m,n,n)
        else:
            # Sparse COO: BDk[:, j] += sum_{(i,j)} val * B[:, i]
            BD = np.zeros((m, n, n), dtype=float)
            for k, (rows, cols, vals) in enumerate(self._coo):
                # group by column for cache friendliness
                # Simple loop is fine when sk is small.
                for i, j, v in zip(rows, cols, vals):
                    BD[k, :, j] += v * B[:, i]

        # H_{k,l} = tr(BDk @ BDl) = sum_{i,j} BDk_{i,j} * BDl_{j,i}
        H = np.einsum("kij,lji->kl", BD, BD, optimize=True)
        # Symmetrize for numerical stability
        H = 0.5 * (H + H.T)
        return H




# ============================================================
# Circulant speedups (cyclic group invariance)
# ============================================================

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


# ============================================================
# Specialized basis: the original tridiagonal C(x) case
# ============================================================

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


# ============================================================
# Phi / gradient / Hessian evaluation
# ============================================================

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


# ============================================================
# Main optimization routine
# ============================================================

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
    return_info=False
):
    """
    Solve the convex problem:
        minimize Phi(x) = -log det(A - C(x))
        subject to A - C(x) ≻ 0
    where C(x) in span(D1,...,Dm) is represented by `basis`.

    Returns (B, C, x) with:
        C = C(x) in S,
        B = (A - C)^{-1} in S_{++},
        and (approximately) tr(B Dk) = 0 for all k.

    Efficiency notes:
      - For structured/sparse basis, implement basis.build_C and basis.trace_with.
      - For Newton, implement basis.hessian_from_B for your special cases
        (e.g., TridiagC_Basis already does this).
      - Otherwise, Newton will form a full Hessian which is only practical when m is small.
        If m is large, prefer method="quasi-newton".
    """
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
        return B, C, x

    if method not in ("gradient-descent", "quasi-newton", "newton"):
        raise ValueError("method must be one of {'gradient-descent','quasi-newton','newton'}.")

    x = np.zeros(m, dtype=float)
    phi, g, H, B, C, M, L = phi_grad_hess_spd(A, x, basis, order=(2 if method == "newton" else 1))

#    min_diag_L = float(np.min(np.diag(L)))
    if isinstance(L, np.ndarray) and L.ndim == 2:
        min_diag_L = float(np.min(np.diag(L)))
    else:
        # circulant/FFT path (no Cholesky)
        min_diag_L = np.nan  # or 1.0, or just skip this diagnostic

    if verbose:
        print(f"...  min(diag(L))={min_diag_L:.3e}")

    if method == "quasi-newton":
        H_BFGS = np.eye(m)

    total_backtracks = 0
    iters_done = 0

    for it in range(max_iter):
        iters_done = it + 1
        g_norm = float(np.linalg.norm(g))
        max_abs_g = float(np.max(np.abs(g)))
        if verbose:
#            lam_min = float(np.min(np.linalg.eigvalsh(M)))
            print(f"iter {it:4d}  phi={phi: .6e}  ||g||={g_norm: .3e}  max|g|={max_abs_g: .3e} ") #  lam_min(A-C)={lam_min: .3e}")

        if max_abs_g < tol:
            if verbose:
                print(f"Converged: max|tr(B Dk)|={max_abs_g:.3e} < tol={tol}")
            break

        # Search direction
        if method == "gradient-descent":
            d = -g

        elif method == "quasi-newton":
            d = -(H_BFGS @ g)

        else:  # newton
            # Hessian may be large to form; allow user control
            if H is None:
                print("None H !!! Aborting !!!")
                return None
#                H = basis.hessian_from_B(B)
#                if H is None:
#                    H = basis.generic_hessian_from_B(B, max_m_for_full=max_m_for_full_hessian)

            # Damped Newton: (H + λI) d = -g
            # Increase damping until Cholesky succeeds.
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
                # fallback
                if verbose:
                    print("  Newton Hessian not SPD even after damping; fallback to -g.")
                d = -g

        # Ensure descent direction
        gTd = float(g @ d)
        if gTd >= 0:
            if verbose:
                print("  Non-descent direction; using -g.")
            d = -g
            gTd = float(g @ d)

        # Backtracking line search
        t = float(initial_step)
        accepted = False
        for bt in range(max_backtracks):
            total_backtracks += 1
            x_try = x + t * d
            try:
                # FAST trial: only check feasibility + phi
                phi_try = phi_only_spd(A, x_try, basis)
            except np.linalg.LinAlgError:
                # infeasible (A - C not SPD)
                t *= backtracking_factor
                continue

            # Armijo condition: phi(x+td) <= phi(x) + alpha t g^T d
            if phi_try <= phi + armijo_alpha * t * gTd:
                accepted = True
                break

            t *= backtracking_factor


        if not accepted:
            if verbose:
                print("Backtracking failed; stopping.")
            break

        # We accepted x_try based on phi-only test.
        # Now compute full quantities ONCE at the accepted point.
        phi_try, g_try, H_try, B_try, C_try, M_try, L_try = phi_grad_hess_spd(
            A, x_try, basis, order=(2 if method == "newton" else 1)
        )

        # BFGS update
        if method == "quasi-newton":
            s = x_try - x
            y_vec = g_try - g
            sy = float(s @ y_vec)
            if sy > 1e-12:
                rho = 1.0 / sy
                I = np.eye(m)
                V = I - rho * np.outer(s, y_vec)
                H_BFGS = V @ H_BFGS @ V.T + rho * np.outer(s, s)

        # Accept
        x, phi, g, B, C, M = x_try, phi_try, g_try, B_try, C_try, M_try
        if method == "newton":
            H = H_try

    # compute a final constraint violation number (max |tr(B Dk)|)
    # works for any basis that implements trace_with
    if hasattr(basis, "trace_with"):
        _g_final = basis.trace_with(B)
        max_trace = float(np.max(np.abs(_g_final))) if _g_final.size else 0.0
    else:
        max_trace = float("nan")

    info = {
        "iters": iters_done,
        "backtracks": total_backtracks,
        "converged": (max_trace < tol) if np.isfinite(max_trace) else False,
        "final_max_abs_trace": max_trace,
    }

    if return_info:
        return B, C, x, info
    return B, C, x

    return B, C, x


# ============================================================
# Example: build a generic sparse basis from COO
# ============================================================

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




# ============================================================
# Dual + group-invariant extensions
# ============================================================


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
    """
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
    """
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
    """
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
    return_info=False
):
    """
    Dual Newton solver (mirrors constrained_decomposition for the primal).

    Solves:
        minimize_B  -log det(B) + tr(A B)
        s.t.         B ≻ 0,   tr(B Dk)=0  (i.e., B ∈ S^\perp)

    Inputs:
      - A: SPD matrix
      - basis: SymBasis spanning S (optional if basis_perp is provided and S is huge)
      - basis_perp: optional SymBasis spanning S^\perp.
        If None, this implementation REQUIRES you to provide it explicitly
        (we do not auto-construct S^\perp from S).

    Output:
      (B, C, y, basis_perp) where:
        B ≻ 0 and B ⟂ S,
        C ∈ S and A ≈ B^{-1} + C
        y are the coordinates of B in basis_perp.
    """
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
            print(f"[dual] iter {it:4d}  psi={psi: .6e}  ||g||={g_norm: .3e}  max|g|={max_abs_g: .3e}")

        if max_abs_g < tol:
            if verbose:
                print(f"[dual] Converged: max|grad|={max_abs_g:.3e} < tol={tol}")
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
                print("[dual] Backtracking failed; stopping.")
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


# ============================================================
# Group invariance utilities
# ============================================================

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
    group: list of permutation vectors or permutation matrices.
    """
    M = np.asarray(M, dtype=float)
    if n is None:
        n = M.shape[0]
    out = np.zeros_like(M)
    for g in group:
        P = _as_perm_matrix(g, n)
        out += P @ M @ P.T
    out /= float(len(group))
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
    **kwargs,
):
    """
    Convenience wrapper:
      1) Optionally symmetrize A under the group (recommended if A is only approximately invariant).
      2) Restrict the search to the G-invariant part of the constraint space S (for C),
         i.e., replace basis by make_group_invariant_basis(basis, group).
      3) Solve either primal or dual.

    solver: "primal" or "dual"
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    A_sym = group_average_conjugation(A, group, n=n)

    basis_G = make_group_invariant_basis(basis, group)

    if solver == "primal":
        B, C, x = constrained_decomposition(
            A=A_sym,
            basis=basis_G,
            method=method,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            **kwargs,
        )
        return B, C, x, basis_G, A_sym

    if solver == "dual":
        B, C, y, basis_perp = constrained_decomposition_dual(
            A=A_sym,
            basis=basis_G,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
            **kwargs,
        )
        return B, C, y, basis_G, basis_perp, A_sym

    raise ValueError("solver must be 'primal' or 'dual'.")



# ============================================================
# Convenience: solve directly in circulant subspace using FFT speedups
# ============================================================

import numpy as np

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


# ============================================================
# Helpers
# ============================================================
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

def time_solve(solve_fn, A, basis, **kwargs):
    t0 = time.perf_counter()
    sol = solve_fn(A, basis, **kwargs)
    t1 = time.perf_counter()
    return sol, (t1 - t0)

# ============================================================
# Unified solvers (single call site style)
# ============================================================
def solve_primal(A, basis, method="newton", **kwargs):
    B, C, x, info = constrained_decomposition(
        A=A,
        basis=basis,
        method=method,
        tol=1e-8,
        max_iter=300,
        verbose=verbose,
        return_info=True,
        **kwargs,
    )
    return {"B": B, "C": C, "x": x, "solver": f"primal-{method}", "info": info}


def solve_dual(A, basis, basis_perp=None):
    # basis may be None for huge S (see small patch to dual solver)
    B, C, y, basis_perp_out, info = constrained_decomposition_dual(
        A=A,
        basis=basis,
        basis_perp=basis_perp,
        tol=1e-8,
        max_iter=300,
        verbose=verbose,
        return_info=True
    )
    return {"B": B, "C": C, "x": y, "solver": "dual", "basis_perp": basis_perp_out, "info": info}



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # ============================================================
    # PyCharm defaults (used when no CLI args override them)
    # ============================================================
    DEFAULT_RUN = {
        "demo1_primal_smallS": True,
        "demo2_dual_antibanded": True,
        "demo3_block_group": True,
        "demo4_banded_A_newton": True,
        "demo5_banded_A_quasi": True,
    }
    DEFAULT_OUTDIR = "demo_outputs"
    DEFAULT_VERBOSE = True

    # ============================================================
    # CLI parsing (overrides defaults when provided)
    # ============================================================
    parser = argparse.ArgumentParser(
        description="Constrained decomposition demos: primal / dual / group-invariant / banded."
    )
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save plots.")
    parser.add_argument("--verbose", action="store_true", help="Verbose solver output.")
    parser.add_argument("--all", action="store_true", help="Run all demos.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Comma-separated list of demos to run. Options: "
             "demo1_primal_smallS,demo2_dual_antibanded,demo3_block_group,"
             "demo4_banded_A_newton,demo5_banded_A_quasi"
    )

    # demo sizes
    parser.add_argument("--n1", type=int, default=40, help="n for demo1 (general A, small S, primal).")
    parser.add_argument("--n2", type=int, default=120, help="n for demo2 (dual on banded S^perp).")
    parser.add_argument("--n3", type=int, default=200, help="n for demo3 (block-permutation group).")
    parser.add_argument("--n4", type=int, default=300, help="n for demo4/5 (banded A + banded S).")

    # parameters
    parser.add_argument("--blocks", type=int, default=5, help="Number of blocks r for demo3.")
    parser.add_argument("--bandwidth4", type=int, default=2, help="Half-bandwidth b for demo4/5 banded A and S.")
    parser.add_argument("--bandwidth2", type=int, default=1, help="Half-bandwidth b for demo2 S^perp (b=1 => tridiag).")

    args = parser.parse_args()

    # Decide which demos to run
    run_flags = dict(DEFAULT_RUN)
    if args.all:
        for k in run_flags:
            run_flags[k] = True
    if args.run is not None:
        for k in run_flags:
            run_flags[k] = False
        chosen = [s.strip() for s in args.run.split(",") if s.strip()]
        for name in chosen:
            if name not in run_flags:
                raise ValueError(f"Unknown demo '{name}'. Valid: {list(run_flags.keys())}")
            run_flags[name] = True

    outdir = args.outdir if args.outdir is not None else DEFAULT_OUTDIR
    os.makedirs(outdir, exist_ok=True)

    # IMPORTANT: solve_primal / solve_dual use the GLOBAL variable `verbose`
    globals()["verbose"] = bool(args.verbose or DEFAULT_VERBOSE)

    # ============================================================
    # Cache demo3 build (so we don't call make_block_fixed_spd twice)
    # ============================================================
    _demo3_cache = {"built": False, "A": None, "basis": None}
    def build_demo3_once():
        if not _demo3_cache["built"]:
            A3, blocks3 = make_block_fixed_spd(args.n3, r=args.blocks, seed=2)
            basis3 = make_block_fixed_basis_offdiag(args.n3, blocks3)
            _demo3_cache["A"] = A3
            _demo3_cache["basis"] = basis3
            _demo3_cache["built"] = True
        return _demo3_cache["A"], _demo3_cache["basis"]

    # ============================================================
    # Cache demo4/5 build (same A,basis for fair Newton vs quasi compare)
    # ============================================================
    _demo45_cache = {"built": False, "A": None, "basis": None}
    def build_demo45_once():
        if not _demo45_cache["built"]:
            A4 = make_banded_spd(args.n4, b=args.bandwidth4, seed=3, diag_boost=8.0)
            basis4 = make_banded_basis(args.n4, b=args.bandwidth4, include_diag=False)
            _demo45_cache["A"] = A4
            _demo45_cache["basis"] = basis4
            _demo45_cache["built"] = True
        return _demo45_cache["A"], _demo45_cache["basis"]

    # ============================================================
    # Demo specs
    # Each build returns (A, basis, solver_kwargs)
    # ============================================================
    demos = [
        {
            "name": "demo1_primal_smallS",
            "title": "General SPD A + small S (few offdiag constraints)  (primal)",
            "build": lambda: (
                make_random_spd(args.n1, seed=0),
                make_offdiag_pair_basis(args.n1, pairs=[(0, 1), (1, 2), (2, 3), (0, 3), (5, 7)]),
                {"method": "newton"},
            ),
            "solve": solve_primal,
            "plot_file": "demo1_primal_smallS.png",
        },
        {
            "name": "demo2_dual_antibanded",
            "title": "General SPD A + anti-banded S (implicit), S^perp banded (small)  (dual)",
            "build": lambda: (
                make_random_spd(args.n2, seed=1),
                None,  # huge S not materialized
                {"basis_perp": make_banded_basis(args.n2, b=args.bandwidth2, include_diag=True)},
            ),
            "solve": solve_dual,
            "plot_file": "demo2_dual_antibanded.png",
        },
        {
            "name": "demo3_block_group",
            "title": "Group-invariant block-permutation A + reduced S^G (block-constant offdiag)  (primal)",
            "build": lambda: (
                build_demo3_once()[0],
                build_demo3_once()[1],
                {"method": "newton"},
            ),
            "solve": solve_primal,
            "plot_file": "demo3_block_group.png",
        },
        {
            "name": "demo4_banded_A_newton",
            "title": "Banded SPD A + banded S (same bandwidth)  (primal NEWTON)",
            "build": lambda: (
                build_demo45_once()[0],
                build_demo45_once()[1],
                {"method": "newton"},
            ),
            "solve": solve_primal,
            "plot_file": "demo4_banded_A_newton.png",
        },
        {
            "name": "demo5_banded_A_quasi",
            "title": "Banded SPD A + banded S (same bandwidth)  (primal QUASI-NEWTON)",
            "build": lambda: (
                build_demo45_once()[0],
                build_demo45_once()[1],
                {"method": "quasi-newton"},
            ),
            "solve": solve_primal,
            "plot_file": "demo5_banded_A_quasi.png",
        },
    ]

    # ============================================================
    # Run loop (ONE solve call site + ONE plot call site)
    # ============================================================
    any_ran = False
    for spec in demos:
        if not run_flags.get(spec["name"], False):
            continue
        any_ran = True

        A, basis, solver_kwargs = spec["build"]()
        # --- print header BEFORE solve so verbose output is contextual
        print("\n" + "=" * 72)
        print(f"[{spec['name']}] {spec['title']}")
        print(f"A_type={spec.get('A_type', '?')}, S_type={spec.get('S_type', '?')}")
        print(f"n={A.shape[0]}, m={getattr(basis, 'm', 'implicit/NA')}, solver={spec.get('solver_tag', '?')}")

        # pass a prefix so iteration lines include the demo name
        solver_kwargs = dict(solver_kwargs)
        solver_kwargs["log_prefix"] = f"[{spec['name']}] "

        sol, elapsed = time_solve(spec["solve"], A, basis, **solver_kwargs)

        B, C = sol["B"], sol["C"]

        # reconstruction diagnostic
        recon = np.linalg.norm(A - (spd_inverse(B) + C), ord="fro")

        # choose basis for trace check and plotting:
        # - if primal: basis is the constraint basis
        # - if dual with basis=None: basis_perp exists in sol and is usable for trace/plotting
        basis_for_check = basis if basis is not None else sol.get("basis_perp", None)

        if basis_for_check is not None and hasattr(basis_for_check, "trace_with"):
            max_trace = float(np.max(np.abs(basis_for_check.trace_with(B))))
        else:
            max_trace = float("nan")

        info = sol.get("info", {})
        iters = info.get("iters", "?")
        backtracks = info.get("backtracks", "?")

        print("\n" + "=" * 72)
        print(f"[{spec['name']}] {spec['title']}")
        print(f"n={A.shape[0]}, m={getattr(basis, 'm', 'implicit/NA')}, solver={sol.get('solver','?')}")
        print(f"time={elapsed:.4f}s, iters={iters}, backtracks={backtracks}, recon_fro={recon:.3e}, max|tr(BDk)|={max_trace:.3e}")

        # plot (use basis_for_check so demo2 doesn't crash)
        plot_path = os.path.join(outdir, spec["plot_file"])
        plot_decomposition_heatmaps(
            A, B, C, basis_for_check,
            filename=plot_path,
            add_title=True
        )
        print(f"saved plot -> {plot_path}")

    if not any_ran:
        print("Nothing selected. Use --all or --run demo1_primal_smallS,...")


