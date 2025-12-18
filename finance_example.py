from constrained_decomposition_utils import *
from constrained_decomposition_general import constrained_decomposition, TridiagC_Basis
import matplotlib.pyplot as plt


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


def compute_value_vs_H(H_vec, n=100):
    n_H = len(H_vec)
    val_vec_markovian = np.zeros(n_H)
    val_vec_general = np.zeros(n_H)
    for i in range(n_H):
        # Build matrix
        A = spd_fractional_BM(n, H=H_vec[i], T=1.0)
        A_inv = spd_inverse(A)
        print("Is A pos.def? ", is_spd(A))
        print("Is A_inv pos.def? ", is_spd(A_inv))

        basis = TridiagC_Basis(n)  # keeps your specialized fast case
        B_newt, C_newt, x_newt = constrained_decomposition(
            A=A_inv,
            basis=basis,
            method="newton",
            tol=1e-6,
            max_iter=500,
            verbose=True
        )

        val_vec_markovian[i] = invest_value_markovian(B_newt, A)
        print("Finished MArkovian, no do general: ")

        A_diff = spd_fractional_BM(n, H=H_vec[i], T=1.0, diff_flag = True)

        val_vec_general[i]  = invest_value_general(A_diff)
        print("Run H=", H_vec[i], " val_general=", val_vec_general[i], " val_markovian=", val_vec_markovian[i])

    return val_vec_markovian, val_vec_general

if __name__ == "__main__":
    # --- Experiment settings ---
    n = 100
    H_vec = np.linspace(0.1, 0.9, 17)   # change resolution as you like

    # --- Run ---
    val_markov, val_general = compute_value_vs_H(H_vec, n=n)

    # --- Plot ---
    plt.figure(figsize=(7, 4))
    plt.plot(H_vec, val_markov, label="Markovian (decomposition)")
    plt.plot(H_vec, val_general, label="General", color="red")
    plt.xlabel("H")
    plt.ylabel("log(Value)")
    plt.title(f"Strategy value vs H (n={n})")
    plt.legend()
    plt.tight_layout()

    out_png = f"value_vs_H_n_{n}.png"
    plt.savefig(out_png, dpi=150)
    plt.show()
    print("Saved:", out_png)
