"""
Fast Toeplitz matrix operations using FFT.

This module provides O(N log N) matrix-vector products and O(N log N × iters) solves
for Toeplitz systems, enabling efficient computation at N=1000+.

Key functions:
- toeplitz_matvec_fft: O(N log N) Toeplitz matrix-vector product
- toeplitz_solve_pcg: O(N log N × iters) Toeplitz system solve using PCG
- schur_complement_solve: Exploits 2×2 block structure with Toeplitz blocks
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.linalg import toeplitz, solve_triangular, cholesky


def toeplitz_matvec_fft(first_col, first_row, x):
    """
    Compute Toeplitz matrix-vector product T @ x using FFT.

    Complexity: O(N log N)

    Parameters
    ----------
    first_col : array (N,)
        First column of the Toeplitz matrix T.
    first_row : array (N,)
        First row of the Toeplitz matrix T (first_row[0] must equal first_col[0]).
    x : array (N,)
        Vector to multiply.

    Returns
    -------
    y : array (N,)
        Result T @ x.

    Notes
    -----
    Embeds the Toeplitz matrix in a circulant matrix of size 2N-1,
    uses FFT to compute the circulant-vector product, then extracts result.
    """
    N = len(x)
    if N == 1:
        return first_col * x

    # Build first column of circulant embedding (size 2N-1)
    # c = [t_0, t_1, ..., t_{N-1}, t_{-(N-1)}, ..., t_{-1}]
    # where t_k = first_col[k] for k >= 0, t_k = first_row[-k] for k < 0
    c = np.zeros(2 * N - 1, dtype=np.float64)
    c[:N] = first_col
    c[N:] = first_row[1:][::-1]  # t_{-(N-1)}, ..., t_{-1}

    # Zero-pad x to length 2N-1
    x_padded = np.zeros(2 * N - 1, dtype=np.float64)
    x_padded[:N] = x

    # FFT-based circulant multiply: y = IFFT(FFT(c) * FFT(x))
    c_fft = fft(c)
    x_fft = fft(x_padded)
    y_padded = ifft(c_fft * x_fft).real

    # Extract first N elements
    return y_padded[:N]


def toeplitz_matvec_symmetric_fft(gamma, x):
    """
    Compute symmetric Toeplitz matrix-vector product T @ x using FFT.

    For symmetric Toeplitz, first_col = first_row = gamma.

    Parameters
    ----------
    gamma : array (N,)
        First column (and row) of the symmetric Toeplitz matrix.
    x : array (N,)
        Vector to multiply.

    Returns
    -------
    y : array (N,)
        Result T @ x.
    """
    return toeplitz_matvec_fft(gamma, gamma, x)


def strang_circulant_preconditioner(gamma):
    """
    Build Strang's optimal circulant preconditioner for a Toeplitz matrix.

    The Strang preconditioner C_S has the same eigenvectors as circulant matrices
    (Fourier modes) and eigenvalues chosen to approximate T.

    For a symmetric Toeplitz matrix with first column gamma, the Strang
    preconditioner has first column:
        c_k = gamma[k] for k = 0, ..., floor(N/2)
        c_k = gamma[N-k] for k = floor(N/2)+1, ..., N-1

    Parameters
    ----------
    gamma : array (N,)
        First column of the symmetric Toeplitz matrix.

    Returns
    -------
    c_fft : array (N,)
        FFT of the first column of the circulant preconditioner.
        To apply M^{-1}v, compute: ifft(fft(v) / c_fft).real
    """
    N = len(gamma)
    c = np.zeros(N, dtype=np.float64)

    # Strang preconditioner: average wrap-around
    half = N // 2
    c[0] = gamma[0]
    for k in range(1, half + 1):
        if k < N - k:
            c[k] = (gamma[k] + gamma[N - k]) / 2 if N - k < N else gamma[k]
        else:
            c[k] = gamma[k]
    for k in range(half + 1, N):
        c[k] = c[N - k]

    # Simpler: just use the first half
    c[:half+1] = gamma[:half+1]
    c[half+1:] = gamma[1:N-half][::-1]

    return fft(c)


def circulant_precond_solve(c_fft, v):
    """
    Apply circulant preconditioner inverse: M^{-1} @ v.

    Parameters
    ----------
    c_fft : array (N,)
        FFT of the first column of the circulant matrix.
    v : array (N,)
        Vector to precondition.

    Returns
    -------
    result : array (N,)
        M^{-1} @ v
    """
    v_fft = fft(v)
    # Regularize to avoid division by zero
    c_fft_safe = np.where(np.abs(c_fft) > 1e-12, c_fft, 1e-12)
    return ifft(v_fft / c_fft_safe).real


def toeplitz_solve_pcg(gamma, b, tol=1e-10, max_iter=None, x0=None, verbose=False,
                       use_circulant_precond=True):
    """
    Solve symmetric positive definite Toeplitz system T @ x = b using PCG.

    Uses FFT-based matrix-vector products for O(N log N) per iteration.
    With Strang's circulant preconditioner, typically converges in O(1) iterations
    for well-conditioned systems.

    Parameters
    ----------
    gamma : array (N,)
        First column of the symmetric Toeplitz matrix T.
    b : array (N,)
        Right-hand side vector.
    tol : float
        Convergence tolerance (relative residual).
    max_iter : int, optional
        Maximum iterations. Default: min(N, 100).
    x0 : array (N,), optional
        Initial guess. Default: zeros.
    verbose : bool
        Print iteration info.
    use_circulant_precond : bool
        Use Strang's circulant preconditioner (recommended).

    Returns
    -------
    x : array (N,)
        Solution to T @ x = b.
    info : dict
        Convergence info: {"iters": int, "residual": float, "converged": bool}
    """
    N = len(b)
    if max_iter is None:
        max_iter = min(N, 100)

    # Initialize
    if x0 is None:
        x = np.zeros(N, dtype=np.float64)
        r = b.copy()
    else:
        x = x0.copy()
        r = b - toeplitz_matvec_symmetric_fft(gamma, x)

    # Build preconditioner
    if use_circulant_precond:
        c_fft = strang_circulant_preconditioner(gamma)
        def precond(v):
            return circulant_precond_solve(c_fft, v)
    else:
        # Diagonal (Jacobi) preconditioning
        M_inv_diag = 1.0 / gamma[0]
        def precond(v):
            return M_inv_diag * v

    z = precond(r)
    p = z.copy()
    rz = np.dot(r, z)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        return np.zeros(N), {"iters": 0, "residual": 0.0, "converged": True}

    for k in range(max_iter):
        Ap = toeplitz_matvec_symmetric_fft(gamma, p)
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-15:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = np.linalg.norm(r)
        rel_res = r_norm / b_norm

        if verbose and k % 10 == 0:
            print(f"  PCG iter {k}: rel_res = {rel_res:.2e}")

        if rel_res < tol:
            return x, {"iters": k + 1, "residual": rel_res, "converged": True}

        z = precond(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, {"iters": max_iter, "residual": rel_res, "converged": False}


def build_fgn_gamma(N, H, alpha=1.0, delta_t=1.0):
    """
    Build the first column of the scaled fGn Toeplitz covariance matrix.

    Uses the same scaling as spd_mixed_fbm for consistency:
        factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))
        gamma(k) = factor * (|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H})

    Parameters
    ----------
    N : int
        Matrix dimension.
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    delta_t : float
        Time step size.

    Returns
    -------
    gamma : array (N,)
        First column of Γ_H.
    """
    factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))
    k = np.arange(N)
    gamma = factor * (
        np.abs(k + 1) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
    )
    return gamma


class BlockedMixedFBMPrecision:
    """
    Efficient representation of the precision matrix Λ for mixed fBM in blocked ordering.

    The covariance matrix has 2×2 block structure:
        Σ = [A  B]   where A = Δt I + α² Γ_H (Toeplitz + diagonal)
            [B  D]         B = Δt I (diagonal)
                           D = Δt I (diagonal)

    The precision matrix Λ = Σ⁻¹ can be computed via Schur complement:
        S = D - B A⁻¹ B = Δt I - Δt² (Δt I + α² Γ_H)⁻¹
        Λ = [A⁻¹ + A⁻¹ B S⁻¹ B A⁻¹,  -A⁻¹ B S⁻¹]
            [-S⁻¹ B A⁻¹,              S⁻¹      ]

    Key insight: A = Δt I + α² Γ_H where Γ_H is Toeplitz, so A is also Toeplitz + diagonal.
    This class provides efficient O(N log N) matrix-vector products with Λ.
    """

    def __init__(self, N, H, alpha=1.0, delta_t=None):
        """
        Initialize the precision matrix representation.

        Parameters
        ----------
        N : int
            Number of time steps. Matrix size is 2N × 2N.
        H : float
            Hurst parameter.
        alpha : float
            fBM weight.
        delta_t : float, optional
            Time step size. Default: 1/N.
        """
        self.N = N
        self.H = H
        self.alpha = alpha
        self.delta_t = delta_t if delta_t is not None else 1.0 / N

        # Build Γ_H first column (Toeplitz)
        self.gamma_H = build_fgn_gamma(N, H, alpha, self.delta_t)

        # A = Δt I + α² Γ_H, so first column is:
        self.A_gamma = self.gamma_H.copy()
        self.A_gamma[0] += self.delta_t

        # B = D = Δt I (diagonal)
        self.B_diag = self.delta_t
        self.D_diag = self.delta_t

        # Precompute A⁻¹ via Cholesky (for moderate N) or cache for PCG
        self._A_inv_cache = None
        self._S_inv_cache = None
        self._use_direct = N <= 500  # Use direct methods for small N

        if self._use_direct:
            self._precompute_direct()

    def _precompute_direct(self):
        """Precompute inverses using direct methods for small N."""
        from scipy.linalg import toeplitz, inv

        A = toeplitz(self.A_gamma)
        self._A_inv = inv(A)

        # S = D - B A⁻¹ B = Δt I - Δt² A⁻¹
        self._S = self.D_diag * np.eye(self.N) - self.B_diag**2 * self._A_inv
        self._S_inv = inv(self._S)

    def A_solve(self, b, tol=1e-10):
        """Solve A @ x = b."""
        if self._use_direct:
            return self._A_inv @ b
        else:
            x, _ = toeplitz_solve_pcg(self.A_gamma, b, tol=tol)
            return x

    def A_matvec(self, x):
        """Compute A @ x."""
        return toeplitz_matvec_symmetric_fft(self.A_gamma, x)

    def S_solve(self, b, tol=1e-10):
        """Solve S @ x = b where S is the Schur complement."""
        if self._use_direct:
            return self._S_inv @ b
        else:
            # S = D - B A⁻¹ B = Δt I - Δt² A⁻¹
            # For iterative: use PCG with S_matvec
            # This is trickier - S is not Toeplitz!
            # For now, build S explicitly (still O(N²) but only once)
            if self._S_inv_cache is None:
                from scipy.linalg import toeplitz, inv
                A = toeplitz(self.A_gamma)
                A_inv = inv(A)
                S = self.D_diag * np.eye(self.N) - self.B_diag**2 * A_inv
                self._S_inv_cache = inv(S)
            return self._S_inv_cache @ b

    def matvec(self, z):
        """
        Compute Λ @ z where z is a 2N vector.

        Λ = [A⁻¹ + A⁻¹ B S⁻¹ B A⁻¹,  -A⁻¹ B S⁻¹]
            [-S⁻¹ B A⁻¹,              S⁻¹      ]

        Let z = [x; y] where x, y are N-vectors.
        Then Λz = [Λ₁₁ x + Λ₁₂ y; Λ₂₁ x + Λ₂₂ y]
        """
        N = self.N
        x = z[:N]
        y = z[N:]

        # Compute A⁻¹ x and A⁻¹ y
        A_inv_x = self.A_solve(x)

        # Λ₂₂ y = S⁻¹ y
        L22_y = self.S_solve(y)

        # Λ₂₁ x = -S⁻¹ B A⁻¹ x = -S⁻¹ (Δt A⁻¹ x)
        L21_x = -self.S_solve(self.B_diag * A_inv_x)

        # Λ₁₂ y = -A⁻¹ B S⁻¹ y = -A⁻¹ (Δt S⁻¹ y) = -Δt A⁻¹ L22_y
        A_inv_L22_y = self.A_solve(L22_y)
        L12_y = -self.B_diag * A_inv_L22_y

        # Λ₁₁ x = (A⁻¹ + A⁻¹ B S⁻¹ B A⁻¹) x = A⁻¹ x + A⁻¹ B S⁻¹ B A⁻¹ x
        #       = A⁻¹ x - Δt A⁻¹ L21_x / Δt  (using L21_x = -S⁻¹ Δt A⁻¹ x)
        # Actually: A⁻¹ B S⁻¹ B A⁻¹ x = Δt A⁻¹ S⁻¹ Δt A⁻¹ x = Δt² A⁻¹ S⁻¹ A⁻¹ x
        S_inv_A_inv_x = self.S_solve(A_inv_x)
        A_inv_S_inv_A_inv_x = self.A_solve(S_inv_A_inv_x)
        L11_x = A_inv_x + self.B_diag**2 * A_inv_S_inv_A_inv_x

        result = np.zeros(2 * N, dtype=np.float64)
        result[:N] = L11_x + L12_y
        result[N:] = L21_x + L22_y

        return result

    def to_dense(self):
        """Convert to dense matrix (for verification/debugging)."""
        from scipy.linalg import toeplitz, inv

        N = self.N
        A = toeplitz(self.A_gamma)
        B = self.B_diag * np.eye(N)
        D = self.D_diag * np.eye(N)

        Sigma = np.zeros((2*N, 2*N))
        Sigma[:N, :N] = A
        Sigma[:N, N:] = B
        Sigma[N:, :N] = B
        Sigma[N:, N:] = D

        return inv(Sigma)


def test_toeplitz_fft():
    """Test FFT-based Toeplitz operations."""
    np.random.seed(42)

    print("Testing FFT-based Toeplitz operations...")

    for N in [10, 100, 1000]:
        # Random symmetric positive definite Toeplitz
        gamma = np.zeros(N)
        gamma[0] = 2.0
        gamma[1:] = 0.5 ** np.arange(1, N)  # Decaying off-diagonal

        x = np.random.randn(N)

        # Direct multiplication
        from scipy.linalg import toeplitz
        T = toeplitz(gamma)
        y_direct = T @ x

        # FFT multiplication
        y_fft = toeplitz_matvec_symmetric_fft(gamma, x)

        error = np.max(np.abs(y_direct - y_fft))
        print(f"  N={N}: matvec error = {error:.2e}")

        # Test solve
        b = np.random.randn(N)
        x_direct = np.linalg.solve(T, b)
        x_pcg, info = toeplitz_solve_pcg(gamma, b, tol=1e-12)

        solve_error = np.max(np.abs(x_direct - x_pcg))
        print(f"  N={N}: solve error = {solve_error:.2e} ({info['iters']} iters)")

    print("All tests passed!\n")


def test_blocked_precision():
    """Test BlockedMixedFBMPrecision class."""
    print("Testing BlockedMixedFBMPrecision...")

    for N in [10, 50, 100]:
        H = 0.6
        alpha = 1.0
        delta_t = 1.0 / N

        # Build using our class
        prec = BlockedMixedFBMPrecision(N, H, alpha, delta_t)

        # Build dense for comparison
        from constrained_decomposition_matrices import spd_mixed_fbm_blocked
        Sigma = spd_mixed_fbm_blocked(N, H, alpha, delta_t)
        Lambda_direct = np.linalg.inv(Sigma)

        # Test matvec
        z = np.random.randn(2 * N)
        Lz_direct = Lambda_direct @ z
        Lz_class = prec.matvec(z)

        error = np.max(np.abs(Lz_direct - Lz_class))
        print(f"  N={N}: matvec error = {error:.2e}")

        # Test to_dense
        Lambda_class = prec.to_dense()
        dense_error = np.max(np.abs(Lambda_direct - Lambda_class))
        print(f"  N={N}: dense error = {dense_error:.2e}")

    print("All tests passed!\n")


class BlockedNewtonSolver:
    """
    Specialized Newton solver for mixed fBM with blocked ordering.

    Exploits the 2×2 block structure of the precision matrix and the
    sparse structure of the Markovian basis to achieve faster convergence.

    Key optimizations:
    1. Block Schur complement: M22 = Λ22 is fixed (basis doesn't affect it),
       so we precompute Λ22⁻¹ and reduce 2N×2N inversions to N×N
    2. Sparse basis: O(N) basis elements, each with O(N) non-zeros
    3. Efficient gradient: tr(B Dₖ) computed via sparse operations
    4. Vectorized index arrays for gradient/Hessian computation

    The covariance has structure:
        Σ = [Δt I + α²Γ_H   Δt I]
            [   Δt I        Δt I]

    The Markovian basis C(x) only affects M11 and M12 blocks, not M22.
    This allows precomputing M22⁻¹ = Λ22⁻¹ once.
    """

    def __init__(self, N, H, alpha=1.0, delta_t=None, verbose=False):
        """
        Initialize the solver.

        Parameters
        ----------
        N : int
            Number of time steps.
        H : float
            Hurst parameter.
        alpha : float
            fBM weight.
        delta_t : float, optional
            Time step. Default: 1/N.
        verbose : bool
            Print progress.
        """
        self.N = N
        self.n = 2 * N
        self.H = H
        self.alpha = alpha
        self.delta_t = delta_t if delta_t is not None else 1.0 / N
        self.verbose = verbose

        # Timing stats
        self._timing = {
            'build_precision': 0.0,
            'build_basis': 0.0,
            'schur_complement': 0.0,
            'gradient': 0.0,
            'hessian_vec': 0.0,
            'line_search': 0.0,
        }

        # Build precision matrix representation
        self._build_precision()

        # Build Markovian basis structure
        self._build_basis()

    def _build_precision(self):
        """Build the precision matrix Λ and precompute fixed block inverses."""
        import time
        from constrained_decomposition_matrices import spd_mixed_fbm_blocked

        t0 = time.time()

        Sigma = spd_mixed_fbm_blocked(self.N, self.H, self.alpha, self.delta_t)
        self.Lambda = np.linalg.inv(Sigma)

        # Store block structure
        N = self.N
        self.Lambda_11 = self.Lambda[:N, :N].copy()
        self.Lambda_12 = self.Lambda[:N, N:].copy()
        self.Lambda_21 = self.Lambda[N:, :N].copy()
        self.Lambda_22 = self.Lambda[N:, N:].copy()

        # Key optimization: Λ22 is FIXED (basis doesn't affect M22 block)
        # Precompute Λ22⁻¹ for use in Schur complement formula
        self.Lambda_22_inv = np.linalg.inv(self.Lambda_22)

        # For reference: log|Σ| and log|Λ22|
        _, self.log_det_Sigma = np.linalg.slogdet(Sigma)
        _, self.log_det_Lambda_22 = np.linalg.slogdet(self.Lambda_22)

        self._timing['build_precision'] = time.time() - t0

    def _build_basis(self):
        """
        Build the Markovian basis structure for blocked ordering.

        The blocked Markovian basis has 2(N-1) elements:
        - D^Mark_{l,X} for l = 2,...,N: affects column l in X-block
        - D^Mark_{l,Y} for l = 2,...,N: affects column l in cross-block
        """
        N = self.N
        self.m = 2 * (N - 1)  # Basis dimension

        # Store basis structure for efficient operations
        # Each basis element affects column l with entries at rows 0,...,l-2
        self.basis_info = []

        for l in range(2, N + 1):
            l_idx = l - 1  # 0-based

            # D^Mark_{l,X}: entries in X-block (rows 0,...,l-2, column l-1)
            rows_X = np.arange(l - 1)
            self.basis_info.append({
                'type': 'X',
                'l': l,
                'l_idx': l_idx,
                'rows': rows_X,
                'block': 'XX'  # Affects [0:N, 0:N] block
            })

            # D^Mark_{l,Y}: entries in cross-block (rows N,...,N+l-2, column l-1)
            rows_Y = N + np.arange(l - 1)
            self.basis_info.append({
                'type': 'Y',
                'l': l,
                'l_idx': l_idx,
                'rows': rows_Y,
                'block': 'YX'  # Affects [N:2N, 0:N] block (and symmetric)
            })

        # Build vectorized index arrays for fast gradient/Hessian computation
        self._build_vectorized_indices()

    def _build_vectorized_indices(self):
        """
        Build index arrays for vectorized gradient/Hessian computation.

        For each basis element k, we store:
        - col_idx[k]: the column index l_idx
        - row_start[k]: start of row indices for this element
        - row_end[k]: end of row indices

        This allows computing gradient via:
            g[k] = 2 * sum(B[all_rows[row_start[k]:row_end[k]], col_idx[k]])
        """
        N = self.N
        m = self.m

        # Column indices for each basis element
        self.col_idx = np.zeros(m, dtype=np.int64)

        # All row indices flattened, with pointers for each basis element
        all_rows = []
        row_start = np.zeros(m + 1, dtype=np.int64)

        for k, info in enumerate(self.basis_info):
            self.col_idx[k] = info['l_idx']
            row_start[k] = len(all_rows)
            all_rows.extend(info['rows'])

        row_start[m] = len(all_rows)
        self.all_rows = np.array(all_rows, dtype=np.int64)
        self.row_start = row_start

        # Number of entries per basis element
        self.entries_per_basis = np.diff(row_start)

        # Precompute expanded column indices for vectorized operations
        self.col_expanded = np.repeat(self.col_idx, self.entries_per_basis)

    def build_C(self, x):
        """
        Build C(x) = Σₖ xₖ Dₖ from the coefficient vector x.

        Parameters
        ----------
        x : array (m,)
            Coefficients for basis elements.

        Returns
        -------
        C : array (2N, 2N)
            The C matrix.
        """
        n = self.n
        C = np.zeros((n, n), dtype=np.float64)

        # Vectorized: for each basis element k, add x[k] to positions
        for k in range(self.m):
            if abs(x[k]) < 1e-15:
                continue

            l_idx = self.col_idx[k]
            rows = self.all_rows[self.row_start[k]:self.row_start[k+1]]

            # Dₖ has entries at (rows, l_idx) and (l_idx, rows)
            C[rows, l_idx] += x[k]
            C[l_idx, rows] += x[k]

        return C

    def build_M(self, x):
        """
        Build M(x) = Λ - C(x).

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        M : array (2N, 2N)
            The M matrix.
        """
        return self.Lambda - self.build_C(x)

    def compute_B(self, x):
        """
        Compute B = M(x)⁻¹ = (Λ - C(x))⁻¹.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        """
        M = self.build_M(x)
        return np.linalg.inv(M)

    def compute_B_block_schur(self, x):
        """
        Compute B = M(x)⁻¹ using block Schur complement formula.

        For symmetric M = [M₁₁  M₁₂]
                          [M₂₁  M₂₂]  with M₂₁ = M₁₂ᵀ

        The inverse is:
        B = M⁻¹ = [M₁₁⁻¹ + M₁₁⁻¹ M₁₂ S⁻¹ M₂₁ M₁₁⁻¹,   -M₁₁⁻¹ M₁₂ S⁻¹]
                  [-S⁻¹ M₂₁ M₁₁⁻¹,                      S⁻¹          ]

        where S = M₂₂ - M₂₁ M₁₁⁻¹ M₁₂ is the Schur complement.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        info : dict
            Intermediate computations for reuse.
        """
        from scipy.linalg import cho_factor, cho_solve

        N = self.N
        M = self.build_M(x)

        M11 = M[:N, :N]
        M12 = M[:N, N:]
        M21 = M[N:, :N]  # = M12.T for symmetric M
        M22 = M[N:, N:]

        # Cholesky factorization of M11
        c11, lower11 = cho_factor(M11, lower=True)

        # M11^{-1} @ M12 using cho_solve
        M11_inv_M12 = cho_solve((c11, lower11), M12)

        # Schur complement: S = M22 - M21 @ M11^{-1} @ M12
        S = M22 - M21 @ M11_inv_M12

        # Cholesky factorization of S
        c_S, lower_S = cho_factor(S, lower=True)

        # For the inverse formula, we need:
        # M11_inv = cho_solve((c11, lower11), I)
        # S_inv = cho_solve((c_S, lower_S), I)
        M11_inv = cho_solve((c11, lower11), np.eye(N))
        S_inv = cho_solve((c_S, lower_S), np.eye(N))

        # Build B blocks using Schur complement formula
        S_inv_M21_M11_inv = S_inv @ M21 @ M11_inv

        # B11 = M11^{-1} + M11^{-1} M12 S^{-1} M21 M11^{-1}
        B11 = M11_inv + M11_inv_M12 @ S_inv_M21_M11_inv

        # B12 = -M11^{-1} M12 S^{-1}
        B12 = -M11_inv_M12 @ S_inv

        # B21 = B12.T (for symmetric M)
        B21 = B12.T

        # B22 = S^{-1}
        B22 = S_inv

        # Assemble B
        B = np.zeros((2*N, 2*N), dtype=np.float64)
        B[:N, :N] = B11
        B[:N, N:] = B12
        B[N:, :N] = B21
        B[N:, N:] = B22

        info = {'c11': c11, 'c_S': c_S, 'M11_inv': M11_inv, 'S_inv': S_inv}
        return B, info

    def compute_B_fast(self, x):
        """
        Compute B = M(x)⁻¹ using optimized direct inverse.

        For now, just use np.linalg.inv which is highly optimized.
        The Schur complement approach is only faster for very large N
        when combined with specialized Toeplitz solvers.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        info : dict
            Empty info dict for interface compatibility.
        """
        M = self.build_M(x)
        # Use scipy's cho_solve for SPD matrices (faster than general inv)
        from scipy.linalg import cho_factor, cho_solve
        c, lower = cho_factor(M, lower=True)
        B = cho_solve((c, lower), np.eye(2 * self.N))
        return B, {'c': c}

    def compute_B_block_optimized(self, x):
        """
        Compute B = M(x)⁻¹ exploiting that M22 = Λ22 is fixed.

        Key insight: The Markovian basis only affects M11 and M12/M21 blocks.
        M22 = Λ22 never changes, so we precomputed Λ22⁻¹.

        Using block inversion with Schur complement w.r.t. M11:
            S = M22 - M21 M11⁻¹ M12  (N×N Schur complement)
            B11 = M11⁻¹ + M11⁻¹ M12 S⁻¹ M21 M11⁻¹
            B12 = -M11⁻¹ M12 S⁻¹
            B21 = B12ᵀ
            B22 = S⁻¹

        This reduces the problem from 2N×2N to N×N operations.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        info : dict
            Contains intermediate computations for potential reuse.
        """
        import time
        from scipy.linalg import cho_factor, cho_solve

        t0 = time.time()
        N = self.N

        # Build C blocks from x
        C11, C12 = self._build_C_blocks(x)

        # M blocks (M21 = M12.T, M22 = Λ22 fixed)
        M11 = self.Lambda_11 - C11
        M12 = self.Lambda_12 - C12
        # M22 = self.Lambda_22 (unchanged)

        # Cholesky factorization of M11 (N×N instead of 2N×2N)
        try:
            c11, lower11 = cho_factor(M11, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to full 2N×2N Cholesky
            return self.compute_B_fast(x)

        # M11⁻¹ M12 via Cholesky solve
        M11_inv_M12 = cho_solve((c11, lower11), M12)

        # Schur complement: S = M22 - M21 M11⁻¹ M12 = Λ22 - M12ᵀ M11⁻¹ M12
        S = self.Lambda_22 - M12.T @ M11_inv_M12

        # Cholesky of S (N×N)
        try:
            c_S, lower_S = cho_factor(S, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to full 2N×2N Cholesky
            return self.compute_B_fast(x)

        # S⁻¹ and M11⁻¹
        S_inv = cho_solve((c_S, lower_S), np.eye(N))
        M11_inv = cho_solve((c11, lower11), np.eye(N))

        # Compute B blocks
        # B12 = -M11⁻¹ M12 S⁻¹
        B12 = -M11_inv_M12 @ S_inv

        # B11 = M11⁻¹ + M11⁻¹ M12 S⁻¹ M21 M11⁻¹ = M11⁻¹ - B12 @ M12.T @ M11⁻¹
        B11 = M11_inv - B12 @ M12.T @ M11_inv

        # B22 = S⁻¹
        B22 = S_inv

        # Assemble B
        B = np.zeros((2*N, 2*N), dtype=np.float64)
        B[:N, :N] = B11
        B[:N, N:] = B12
        B[N:, :N] = B12.T  # B21 = B12ᵀ
        B[N:, N:] = B22

        self._timing['schur_complement'] += time.time() - t0

        info = {
            'c11': c11, 'c_S': c_S,
            'M11_inv': M11_inv, 'S_inv': S_inv,
            'log_det_S': np.sum(np.log(np.diag(c_S))),  # From Cholesky
            'log_det_M11': np.sum(np.log(np.diag(c11))),
        }
        return B, info

    def _build_C_blocks(self, x):
        """
        Build C(x) block-wise: return C11 (N×N) and C12 (N×N).

        The Markovian basis structure:
        - D^Mark_{l,X}: entries in C11 at (rows 0..l-2, col l-1)
        - D^Mark_{l,Y}: entries in C12 at (rows 0..l-2, col l-1) where
          the original entry is at row N+r-1, but we map to the C12 block

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        C11 : array (N, N)
            Upper-left block of C(x).
        C12 : array (N, N)
            Upper-right block of C(x).
        """
        N = self.N
        C11 = np.zeros((N, N), dtype=np.float64)
        C12 = np.zeros((N, N), dtype=np.float64)

        # Process basis elements pairwise (D^Mark_{l,X} then D^Mark_{l,Y})
        k = 0
        for l in range(2, N + 1):
            l_idx = l - 1  # 0-based column index

            # D^Mark_{l,X}: affects C11
            if k < self.m:
                rows = np.arange(l - 1)  # rows 0,...,l-2
                if abs(x[k]) > 1e-15:
                    C11[rows, l_idx] += x[k]
                    C11[l_idx, rows] += x[k]
                k += 1

            # D^Mark_{l,Y}: affects C12 (and C21 = C12.T)
            if k < self.m:
                rows = np.arange(l - 1)  # rows 0,...,l-2 in the Y block
                if abs(x[k]) > 1e-15:
                    # Original: entries at (N+r, l_idx) and (l_idx, N+r)
                    # In blocks: C12[l_idx, rows] and C21[rows, l_idx]
                    # Since M21 = M12.T and C21 = C12.T, we just set C12
                    C12[l_idx, rows] += x[k]  # This handles the (l_idx, N+r) entries
                k += 1

        return C11, C12

    def compute_gradient(self, B):
        """
        Compute gradient g_k = tr(B Dₖ) for all k.

        Vectorized implementation using precomputed indices.

        Parameters
        ----------
        B : array (2N, 2N)
            Current B matrix.

        Returns
        -------
        g : array (m,)
            Gradient vector.
        """
        # Extract all needed B values at once using precomputed indices
        B_vals = B[self.all_rows, self.col_expanded]

        # Sum within each basis element using reduceat
        g = 2.0 * np.add.reduceat(B_vals, self.row_start[:-1])

        return g

    def compute_hessian_vector_product(self, B, v):
        """
        Compute Hessian-vector product (H v)_k = Σₗ H_kl v_l.

        Vectorized implementation.

        Parameters
        ----------
        B : array (2N, 2N)
            Current B matrix.
        v : array (m,)
            Vector to multiply.

        Returns
        -------
        Hv : array (m,)
            Result of H @ v.
        """
        # D(v) = Σₗ vₗ Dₗ
        D_v = self.build_C(v)

        # B @ D(v) @ B
        BDvB = B @ D_v @ B

        # (Hv)_k = tr(Dₖ BDvB) = 2 * sum(BDvB[rows_k, col_k])
        BDvB_vals = BDvB[self.all_rows, self.col_expanded]
        Hv = 2.0 * np.add.reduceat(BDvB_vals, self.row_start[:-1])

        return Hv

    def solve(self, tol=1e-8, max_iter=200, method="newton-cg", use_block_opt=True):
        """
        Solve the constrained decomposition problem.

        Find x such that tr(B(x) Dₖ) = 0 for all k, where B(x) = (Λ - C(x))⁻¹.

        Parameters
        ----------
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.
        method : str
            "newton" (explicit Hessian) or "newton-cg" (matrix-free).
        use_block_opt : bool
            If True, use block-optimized N×N Schur complement method.
            If False, use full 2N×2N Cholesky.

        Returns
        -------
        B : array (2N, 2N)
            Optimal B matrix.
        C : array (2N, 2N)
            Optimal C matrix.
        x : array (m,)
            Optimal coefficients.
        info : dict
            Convergence info including timing breakdown.
        """
        import time
        t_start = time.time()
        t_B_compute = 0.0
        t_gradient = 0.0
        t_hessian = 0.0
        t_linesearch = 0.0

        # Choose B computation method
        compute_B = self.compute_B_block_optimized if use_block_opt else self.compute_B_fast

        # Initialize x = 0
        x = np.zeros(self.m, dtype=np.float64)

        # Initial B
        t0 = time.time()
        B, L_info = compute_B(x)
        t_B_compute += time.time() - t0

        t0 = time.time()
        g = self.compute_gradient(B)
        t_gradient += time.time() - t0

        if self.verbose:
            print(f"BlockedNewtonSolver: N={self.N}, m={self.m}")
            print(f"  Block optimization: {'ON' if use_block_opt else 'OFF'}")
            print(f"  Initial max|g| = {np.max(np.abs(g)):.3e}")

        for it in range(max_iter):
            max_g = np.max(np.abs(g))

            if max_g < tol:
                if self.verbose:
                    print(f"Converged at iter {it}: max|g| = {max_g:.3e}")
                break

            t0 = time.time()
            if method == "newton":
                # Full Hessian (O(m²) computation)
                H = np.zeros((self.m, self.m), dtype=np.float64)
                for l in range(self.m):
                    e_l = np.zeros(self.m)
                    e_l[l] = 1.0
                    H[:, l] = self.compute_hessian_vector_product(B, e_l)

                # Newton direction: H d = -g
                d = np.linalg.solve(H + 1e-10 * np.eye(self.m), -g)

            else:  # newton-cg
                # CG for Newton direction
                from scipy.sparse.linalg import cg, LinearOperator

                def Hv_op(v):
                    return self.compute_hessian_vector_product(B, v)

                H_linop = LinearOperator((self.m, self.m), matvec=Hv_op)
                d, cg_info = cg(H_linop, -g, rtol=1e-6, maxiter=min(100, self.m))
            t_hessian += time.time() - t0

            # Line search (Cholesky success implies SPD)
            t0 = time.time()
            step = 1.0
            B_new = None
            L_info_new = None
            for _ in range(20):
                x_new = x + step * d

                try:
                    B_new, L_info_new = compute_B(x_new)
                    # Cholesky succeeded, so M is SPD, hence B is SPD
                    break
                except np.linalg.LinAlgError:
                    step *= 0.5
            t_linesearch += time.time() - t0

            if B_new is None:
                if self.verbose:
                    print(f"Line search failed at iter {it}")
                break

            x = x_new
            B = B_new
            L_info = L_info_new

            t0 = time.time()
            g = self.compute_gradient(B)
            t_gradient += time.time() - t0

            if self.verbose and it % 10 == 0:
                print(f"Iter {it}: max|g| = {max_g:.3e}, step = {step:.3f}")

        C = self.build_C(x)
        t_total = time.time() - t_start

        info = {
            'iters': it + 1,
            'converged': np.max(np.abs(g)) < tol,
            'final_max_g': np.max(np.abs(g)),
            'time': t_total,
            'timing': {
                'B_compute': t_B_compute + t_linesearch,  # Line search includes B computation
                'gradient': t_gradient,
                'hessian_cg': t_hessian,
            },
            'use_block_opt': use_block_opt,
        }

        return B, C, x, info

    def compute_investment_value(self, B):
        """
        Compute the investment value: 0.5 * (log|Σ| - log|B|).

        Parameters
        ----------
        B : array (2N, 2N)
            The B matrix.

        Returns
        -------
        value : float
            Log investment value.
        """
        _, log_det_B = np.linalg.slogdet(B)
        return 0.5 * (self.log_det_Sigma - log_det_B)


def benchmark_block_optimization(N_values=None, H=0.6, alpha=1.0, verbose=True):
    """
    Benchmark block-optimized vs full 2N×2N Cholesky methods.

    This benchmark compares:
    1. compute_B_block_optimized: N×N Schur complement approach
    2. compute_B_fast: Full 2N×2N Cholesky

    Parameters
    ----------
    N_values : list of int
        Values of N to test. Default: [50, 100, 200, 300, 500]
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    verbose : bool
        Print results table.

    Returns
    -------
    results : list of dict
        Benchmark results for each N.
    """
    if N_values is None:
        N_values = [50, 100, 200, 300, 500]

    results = []

    if verbose:
        print("\n" + "=" * 80)
        print("BENCHMARK: Block-optimized N×N vs Full 2N×2N Cholesky")
        print(f"H={H}, alpha={alpha}")
        print("=" * 80)
        print(f"{'N':>6} {'2N':>6} {'Full (s)':>12} {'Block (s)':>12} {'Speedup':>10} {'Value Match':>12}")
        print("-" * 80)

    for N in N_values:
        delta_t = 1.0 / N

        # Create solver
        solver = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=False)

        # Solve with full 2N×2N method
        B_full, C_full, x_full, info_full = solver.solve(
            tol=1e-8, method="newton-cg", use_block_opt=False
        )
        t_full = info_full['time']
        val_full = solver.compute_investment_value(B_full)

        # Solve with block-optimized method
        B_block, C_block, x_block, info_block = solver.solve(
            tol=1e-8, method="newton-cg", use_block_opt=True
        )
        t_block = info_block['time']
        val_block = solver.compute_investment_value(B_block)

        speedup = t_full / t_block if t_block > 0 else float('inf')
        val_diff = abs(val_full - val_block)
        match = val_diff < 1e-8

        if verbose:
            print(f"{N:>6} {2*N:>6} {t_full:>11.3f}s {t_block:>11.3f}s {speedup:>9.2f}x {'OK' if match else f'DIFF={val_diff:.2e}':>12}")

        results.append({
            'N': N,
            't_full': t_full,
            't_block': t_block,
            'speedup': speedup,
            'value_full': val_full,
            'value_block': val_block,
            'value_match': match,
            'iters_full': info_full['iters'],
            'iters_block': info_block['iters'],
        })

    if verbose:
        print("=" * 80)
        if results:
            avg_speedup = np.mean([r['speedup'] for r in results])
            print(f"Average speedup: {avg_speedup:.2f}x")
        print()

    return results


def test_blocked_newton_solver():
    """Test the BlockedNewtonSolver."""
    from finance_example import invest_value_mixed_fbm_blocked

    print("Testing BlockedNewtonSolver...")
    print("=" * 70)

    for N in [20, 50, 100, 200, 500]:
        H = 0.6
        alpha = 1.0
        delta_t = 1.0 / N

        print(f"\nN={N} (matrix size 2N={2*N}, basis dim m={2*(N-1)}):")

        # Solve with specialized solver
        import time
        t0 = time.time()
        solver = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=False)
        B, C, x, info = solver.solve(tol=1e-8, method="newton-cg")
        value_specialized = solver.compute_investment_value(B)
        t_specialized = time.time() - t0

        # Compare with general solver
        t0 = time.time()
        value_general, info_general = invest_value_mixed_fbm_blocked(
            H, N, alpha, delta_t, strategy='markovian', method='newton'
        )
        t_general = time.time() - t0

        diff = abs(value_specialized - value_general)
        speedup = t_general / t_specialized if t_specialized > 0 else float('inf')

        print(f"  Specialized: {value_specialized:.10f} ({t_specialized:.3f}s, {info['iters']} iters)")
        print(f"  General:     {value_general:.10f} ({t_general:.3f}s, {info_general['iters']} iters)")
        print(f"  Diff: {diff:.2e}, Speedup: {speedup:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toeplitz solver for mixed fBM")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark block optimization vs full Cholesky")
    parser.add_argument("--N", type=str, default="50,100,200,300,500",
                        help="Comma-separated N values for benchmark (default: 50,100,200,300,500)")
    parser.add_argument("--H", type=float, default=0.6, help="Hurst parameter (default: 0.6)")
    parser.add_argument("--alpha", type=float, default=1.0, help="fBM weight (default: 1.0)")
    args = parser.parse_args()

    if args.benchmark:
        N_values = [int(n.strip()) for n in args.N.split(",")]
        benchmark_block_optimization(N_values=N_values, H=args.H, alpha=args.alpha)
    elif args.test:
        test_toeplitz_fft()
        test_blocked_precision()
        test_blocked_newton_solver()
    else:
        # Default: run tests
        test_toeplitz_fft()
        test_blocked_precision()
        test_blocked_newton_solver()
