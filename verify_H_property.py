import numpy as np
import time
import argparse
from fast_toeplitz_solver import fast_toeplitz_log_det_inv_diag

def get_spd_sum_fbm_first_col(n: int, H: float, alpha: float, nugget: float = 1e-9):
    """
    Generates the first column of the spd_sum_fbm covariance matrix.
    This is sufficient to define the entire symmetric Toeplitz matrix.
    """
    c = np.zeros(n)
    
    # fBM increment covariance factor: alpha^2 / (2 * n^{2H})
    factor = (alpha ** 2) / (2.0 * (n ** (2.0 * H)))
    
    i_vals = np.arange(n, dtype=float)
    
    # fBM increment covariance part
    fbm_cov = (
        np.abs(i_vals + 1) ** (2.0 * H)
        + np.abs(i_vals - 1) ** (2.0 * H)
        - 2.0 * np.abs(i_vals) ** (2.0 * H)
    )
    c = factor * fbm_cov

    # Add BM variance on diagonal (to the first element of the first column)
    c[0] += (alpha ** 2) / n
    
    # Add a small stabilizing nugget for numerical stability
    c[0] += nugget
    
    return c

def main():
    parser = argparse.ArgumentParser(description="Verify mixed fBM value property vs. n and H using a fast Toeplitz solver.")
    parser.add_argument("--n_values", type=int, nargs='+', default=[100, 200, 500, 1000], help="List of n values to test.")
    parser.add_argument("--H_values", type=float, nargs='+', default=[0.7, 0.8], help="List of H values to test.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for spd_sum_fbm.")
    parser.add_argument("--nugget", type=float, default=1e-9, help="Small diagonal stabilizer to add for numerical stability.")
    args = parser.parse_args()

    print(f"--- Verification for mixed_fbm_sum strategy value ---")
    print(f"Testing n = {args.n_values} and H = {args.H_values}")
    print(f"Using O(n^2) fast Toeplitz solver with a nugget of {args.nugget}.")
    print("-" * 60)

    results = {}

    for H in args.H_values:
        results[H] = []
        print(f"Processing for H = {H:.2f}")
        for n in args.n_values:
            print(f"  n = {n:<5} ... ", end="", flush=True)
            start_time = time.time()
            
            try:
                # 1. Generate only the first column of the Toeplitz matrix
                first_col = get_spd_sum_fbm_first_col(n=n, H=H, alpha=args.alpha, nugget=args.nugget)
                
                # 2. Use the fast solver to get log-determinant and inverse diagonal
                log_det_gamma, diag_gamma_inv = fast_toeplitz_log_det_inv_diag(first_col)
                
                # 3. Compute the investment value using the corrected, comparable formula
                if np.any(diag_gamma_inv <= 0):
                    log_value = -np.inf  # Solver failed or matrix not SPD
                else:
                    # Use the formula from invest_value_general for a comparable value
                    # log(Value) = 0.5 * (log|A| + sum(log(diag(A^{-1}))))
                    log_value = 0.5 * (log_det_gamma + np.sum(np.log(diag_gamma_inv)))

                duration = time.time() - start_time
                results[H].append({'n': n, 'value': log_value, 'time': duration})
                
                print(f"log(Value) = {log_value:<12.6f} (took {duration:.4f}s)")

            except Exception as e:
                duration = time.time() - start_time
                print(f"FAILED in {duration:.4f}s. Error: {e}")
                results[H].append({'n': n, 'value': 'ERROR', 'time': duration})
        print("-" * 60)
        
    print("--- Summary of Results ---")
    for H, res_list in results.items():
        print(f"For H = {H:.2f}:")
        for res in res_list:
            print(f"  n={res['n']:<5} -> Value={res['value'] if isinstance(res['value'], str) else f'{res['value']:.4f}':<10} (Time: {res['time']:.4f}s)")

if __name__ == "__main__":
    main()
