import numpy as np
from adjacency_matrix import construct_adjacency_matrix
from preliminary_check import check_isotropicity_same_charge
from eigenvalues import eigenvalues, find_eigval
from read_params import read_params
from lindbladian import construct_lindbladian
from eigenmodes import eigensystem_adjacency
import os
from time import time
from tqdm import tqdm



def diagonalize_lindbladian(Hamiltonian, jump_operators, decay_rates=None):
    result, charges = check_isotropicity_same_charge(Hamiltonian, jump_operators)

    N = Hamiltonian.shape[0]

    # Precompute total iterations
    total_iters = sum(
        max(0, min(N, N - l) - max(0, -l))
        for l in range(-N + 1, N)
    )

    eigvecs = []
    eigvals = []

    with tqdm(total=total_iters, desc="Diagonalizing sectors") as pbar:
        for l in range(-N + 1, N):
            m_min = max(0, -l)
            m_max = min(N, N - l)

            for m in range(m_min, m_max):
                lambda_eff, R, L = eigensystem_adjacency(
                    Hamiltonian, jump_operators, m, l, decay_rates
                )

                eigvecs.append((R, L))
                eigvals.append(lambda_eff)

                pbar.update(1)
                
    return np.array(eigvals), eigvecs


def plot_eigvals(eigvals, plot_spectrum):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(eigvals.real, eigvals.imag, color='blue', marker='o')
    plt.title("Spectrum of the Lindbladian")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.savefig(f"outputs/lindbladian_spectrum_{np.random.randint(1e6)}.png")


if __name__ == "__main__":
    print("Reading parameters...")

    params = read_params("input_parameters.dat")
    N = params["N"]
    Hamiltonian = params["hamil"]
    jump_operators = params["jump_ops"]
    decay_rates = params["decay_rates"]

    save_lindbladian = params["save_lindbladian"]
    save_eigenvalues = params["save_eigenvalues"]
    save_eigenvectors = params["save_eigenvectors"]
    plot_spectrum = params["plot_spectrum"]
    save_adjacency_matrix = params["save_adjacency_matrix"]
    adj_mat_mode_to_save = params["adj_mat_mode_to_save"]
    clean_outputs = params["clean_outputs"]

    print("Parameters loaded successfully.")
    
    print("Checking isotropicity and charge structure...")
    result, charges = check_isotropicity_same_charge(Hamiltonian, jump_operators)
    print("Isotropicity check passed:", result)
    print("Charges of jump operators:", charges)

    print("Diagonalizing the Lindbladian... ")
    start_time = time()
    eigvals, eigvecs = diagonalize_lindbladian(Hamiltonian, jump_operators, decay_rates)
    end_time = time()
    print(f"Diagonalization completed in {end_time - start_time:.2f} seconds.")

    print("Saving results...")
    os.makedirs("outputs", exist_ok=True)
    if clean_outputs:
        for f in os.listdir("outputs"):
            os.remove(os.path.join("outputs", f))

    if save_lindbladian:
        lindbladian = construct_lindbladian(Hamiltonian, jump_operators)
        np.save("outputs/lindbladian.npy", lindbladian)
    if save_eigenvalues:
        np.save("outputs/eigenvalues.npy", eigvals)
    if save_eigenvectors:
        np.save("outputs/eigenvectors.npy", eigvecs)
    if plot_spectrum:
        plot_eigvals(eigvals, plot_spectrum)
    if save_adjacency_matrix:
        m,l = adj_mat_mode_to_save
        A = construct_adjacency_matrix(Hamiltonian, jump_operators, m, l, decay_rates)
        np.save(f"outputs/adjacency_matrix_m{m}_l{l}.npy", A)
    print("All results saved successfully.")




    


