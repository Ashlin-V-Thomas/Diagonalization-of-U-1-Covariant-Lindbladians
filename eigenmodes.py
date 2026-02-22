import numpy as np
from adjacency_matrix import construct_adjacency_matrix
from eigenvalues import find_eigval
from preliminary_check import extract_charges

def eigensystem_adjacency(Hamiltonian, jump_operators, m, l, decay_rates=None):
    N = Hamiltonian.shape[0]
    if decay_rates is None:
        decay_rates = np.ones(len(jump_operators))

    A = construct_adjacency_matrix(Hamiltonian, jump_operators, m, l, decay_rates)

    I = np.eye(N, dtype=complex)

    # e_m picks the normalization
    e = np.zeros(N, dtype=complex)
    e[m] = 1.0

    sign_charge = np.sign(extract_charges(jump_operators)[0])

    if sign_charge > 0:
        x = np.linalg.solve(I - np.tril(A), e)
        y = np.linalg.solve(I - np.triu(A), e)
    else:
        x = np.linalg.solve(I - np.triu(A), e)
        y = np.linalg.solve(I - np.tril(A), e)

    lambda_eff = find_eigval(Hamiltonian, jump_operators, m, l, decay_rates)

    
    R = np.zeros((N, N), dtype=complex)
    for k in range(np.max([0, -l]), np.min([N, N-l])):
        R[k + l, k] = x[k]

    # Build left operator L = Σ_k y[k] |k><k+l|
    L = np.zeros((N, N), dtype=complex)
    for k in range(np.max([0, -l]), np.min([N, N-l])):
        L[k+l, k] = y[k]

    return lambda_eff, R, L


if __name__ == "__main__":
    from read_params import read_params
    params = read_params("input_parameters.dat")
    print("Loaded parameters successfully.")
    lambda_eff, R, L = eigensystem_adjacency(params["hamil"], params["jump_ops"], m=0, l=0, decay_rates=params["decay_rates"])
    print("The m=0, l=0 eigenmode:")
    print("Effective eigenvalue:", lambda_eff)
    print("Right eigenvector R:\n", R)
    print("Left eigenvector L:\n", L)