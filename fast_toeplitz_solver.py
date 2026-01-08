import numpy as np

def fast_toeplitz_log_det_inv_diag(c):
    """
    Computes log-determinant and inverse diagonal of a symmetric Toeplitz matrix.

    This implementation uses the Levinson-Durbin recursion to find the predictor
    coefficients and then uses the O(n^2) Trench's algorithm recurrence
    to construct the inverse matrix.

    Args:
        c (np.ndarray): The first column of the symmetric Toeplitz matrix,
                        [t_0, t_1, ..., t_{n-1}].

    Returns:
        log_det (float): The natural logarithm of the determinant of the matrix.
        inv_diag (np.ndarray): A vector containing the diagonal elements of the inverse matrix.
    """
    n = len(c)
    if n == 0:
        return 0.0, np.array([])

    if c[0] <= 0:
        raise ValueError(f"Matrix is not positive definite (c[0] = {c[0]} <= 0).")

    # Step 1: Levinson-Durbin recursion
    P = c[0]
    log_det = np.log(P)
    a = np.array([1.0])

    for k in range(1, n):
        kappa_num = c[k] + np.dot(c[1:k], a[k-1:0:-1])
        kappa = -kappa_num / P

        a_old = a
        a = np.concatenate((a, [0]))
        a += kappa * np.concatenate(([0], a_old[::-1]))

        P_new = P * (1.0 - kappa * kappa)
        if P_new <= 0:
            raise ValueError(f"Matrix not positive definite. Failed at step k={k} with |kappa|={abs(kappa)} >= 1.")
        P = P_new
        log_det += np.log(P)

    # Step 2: Construct the inverse using Trench's algorithm O(n^2) recurrence
    inv_T = np.zeros((n, n))
    
    # Final prediction error power
    P_final = P 
    if P_final == 0:
        raise ValueError("Cannot invert singular matrix.")

    # From the final predictor polynomial 'a' (phi), construct the inverse
    phi = a
    
    inv_T[0, 0] = 1.0 / P_final
    for i in range(1, n):
        inv_T[0, i] = (1.0 / P_final) * phi[n-i]
        inv_T[i, 0] = inv_T[0, i]

    # Use the recurrence B_ij = B_{i-1,j-1} + (1/P)*(a_i*a_j - a_{n-j}*a_{n-i})
    # Note: my 'a' is a_k, but the formula uses a_i which is confusing.
    # Let's use v_i = a_i, u_i = a_{n-i}.
    v = phi
    u = phi[::-1]

    for i in range(1, n):
        for j in range(1, n):
            inv_T[i, j] = inv_T[i - 1, j - 1] + (1.0 / P_final) * (v[i] * v[j] - u[n-i] * u[n-j])

    return log_det, np.diag(inv_T)
