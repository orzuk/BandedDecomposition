import numpy as np


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


def constrained_decomposition(A, allow_indefinite_B=False,
                              tol=1e-8, max_iter=500,
                              initial_step=1.0, backtracking_factor=0.5,
                              armijo_alpha=1e-4, verbose=False):
    """
    Compute B, C, x such that A ≈ B^{-1} + C, with C satisfying the
    tridiagonal structure and B_{kk} = B_{k,k+1} approximately.

    Parameters
    ----------
    A : (n,n) ndarray
        Symmetric positive-definite matrix.
    allow_indefinite_B : bool, default False
        If False: enforce A - C(x) SPD and use convex formulation
        (unique solution).
        If True: allow A - C(x) to be any invertible symmetric matrix and
        minimize -log|det(A - C(x))|; B may be indefinite.
    tol : float
        Tolerance on gradient norm.
    max_iter : int
        Maximum number of gradient descent iterations.
    initial_step : float
        Initial step size for backtracking line search.
    backtracking_factor : float
        Factor in (0,1) by which step size is multiplied in backtracking.
    armijo_alpha_
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
        return B, C, x                 # <-- return here for n=1

    x = np.zeros(n - 1, dtype=float)

    if allow_indefinite_B:
        phi, grad, B, C, M = phi_and_grad_general(A, x)
    else:
        phi, grad, B, C, M = phi_and_grad_spd(A, x)

    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        if verbose:
            print(f"iter {it}, phi={phi:.6e}, ||grad||={grad_norm:.3e}")
        if grad_norm < tol:
            break

        d = -grad
        t = initial_step
        success = False

        for _ in range(50):
            x_try = x + t * d
            try:
                if allow_indefinite_B:
                    phi_try, grad_try, B_try, C_try, M_try = phi_and_grad_general(A, x_try)
                else:
                    C_tmp = build_C_from_x(x_try)
                    M_tmp = A - C_tmp
                    if not is_spd(M_tmp):
                        t *= backtracking_factor
                        continue
                    phi_try, grad_try, B_try, C_try, M_try = phi_and_grad_spd(A, x_try)
            except np.linalg.LinAlgError:
                t *= backtracking_factor
                continue

            if phi_try <= phi - armijo_alpha * t * grad_norm**2:
                success = True
                x = x_try
                phi, grad, B, C, M = phi_try, grad_try, B_try, C_try, M_try
                break
            else:
                t *= backtracking_factor

        if not success:
            if verbose:
                print("Backtracking failed to find a better point; stopping.")
            break

    # >>> THIS MUST EXIST <<<
    return B, C, x                     # <-- final return



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    random_A = False # True # False
    np.random.seed(0)

    # Construct a random SPD matrix A
    if random_A:
        n = 14
        M = np.random.randn(n, n)
        A = M @ M.T + n * np.eye(n)  # ensures SPD
    else:
        n = 7
        A = np.array([
            [6, 2, 1],
            [2, 5, 2],
            [1, 2, 4]
        ], dtype=float)
        A = np.eye(n)

    print("A =")
    print(A)

    # Compute decomposition with B SPD
    B, C, x = constrained_decomposition(A, allow_indefinite_B=False,
                                        tol=1e-10, verbose=True)

    print("\nDecomposition results (SPD B):")
    print("x =", x)
    print("C =")
    print(C)
    print("B =")
    print(B)

    # Check A ≈ B^{-1} + C
    A_reconstructed = np.linalg.inv(B) + C
    print("\nA_reconstructed =")
    print(A_reconstructed)
    print("\nReconstruction error (Frobenius norm):",
          np.linalg.norm(A - A_reconstructed, ord="fro"))

    # Optionally, try general symmetric B mode
    B_gen, C_gen, x_gen = constrained_decomposition(
        A, allow_indefinite_B=True, tol=1e-10, verbose=False
    )
    A_reconstructed_gen = np.linalg.inv(B_gen) + C_gen
    print("\nGeneral symmetric B mode:")

    err = np.linalg.norm(A - A_reconstructed_gen, ord="fro")
    print(f"\nReconstruction error (Frobenius): {err:.3e}")
