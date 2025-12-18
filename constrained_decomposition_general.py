
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
    def generic_hessian_from_B(self, B, max_m_for_full=500):
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
        return phi, g, None, B, C, M

    H = basis.hessian_from_B(B)
    if H is None:
        H = basis.generic_hessian_from_B(B)

    return phi, g, H, B, C, M


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
    phi, g, H, B, C, M = phi_grad_hess_spd(A, x, basis, order=(2 if method == "newton" else 1))

    if method == "quasi-newton":
        H_BFGS = np.eye(m)

    for it in range(max_iter):
        g_norm = float(np.linalg.norm(g))
        max_abs_g = float(np.max(np.abs(g)))
        if verbose:
            lam_min = float(np.min(np.linalg.eigvalsh(M)))
            print(f"iter {it:4d}  phi={phi: .6e}  ||g||={g_norm: .3e}  max|g|={max_abs_g: .3e}  lam_min(A-C)={lam_min: .3e}")

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
                H = basis.hessian_from_B(B)
                if H is None:
                    H = basis.generic_hessian_from_B(B, max_m_for_full=max_m_for_full_hessian)

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
            x_try = x + t * d
            try:
                phi_try, g_try, H_try, B_try, C_try, M_try = phi_grad_hess_spd(
                    A, x_try, basis, order=(2 if method == "newton" else 1)
                )
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


if __name__ == "__main__":
    # Demo: recover the old specialized behavior via TridiagC_Basis.
    np.set_printoptions(precision=3, suppress=True)

    n = 10
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)

    # Use the special basis (fast C(x), fast grad, fast Hessian)
    basis = TridiagC_Basis(n)

    B, C, x = constrained_decomposition(
        A=A,
        basis=basis,
        method="newton",
        tol=1e-8,
        max_iter=200,
        verbose=True,
    )

    # Check trace constraints (here: g_k = Bkk - Bk,k+1)
    g = basis.trace_with(B)
    print("\nmax |tr(B Dk)| =", np.max(np.abs(g)))
    print("reconstruction error ||A - (B^{-1}+C)||_F =",
          np.linalg.norm(A - (spd_inverse(B) + C), ord="fro"))
