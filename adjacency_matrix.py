import numpy as np
from preliminary_check import check_isotropicity_same_charge
from eigenvalues import find_eigval

def construct_adjacency_matrix(Hamiltonian, jump_operators, m, l, decay_rates=None):
    N = Hamiltonian.shape[0]
    if decay_rates is None:
        decay_rates = np.ones(len(jump_operators))

    # Check isotropicity and obtain charges
    result, charges = check_isotropicity_same_charge(Hamiltonian, jump_operators)
    if not result:
        raise ValueError("The system is not U(1) isotropic with jump operators of the same charge sign.")
    
    sign_charge = np.sign(charges[0])

    # Group by charge
    charge_dict = {}
    for j in range(len(jump_operators)):
        q_j = charges[j]
        if q_j not in charge_dict:
            charge_dict[q_j] = []
        charge_dict[q_j].append((jump_operators[j], decay_rates[j]))

    # Allocate adjacency matrix
    A = np.zeros((N, N), dtype=complex)

    for k in range(N):
        for q_j in charge_dict.keys():

            weight = 0.0 + 0j
            for J, gamma in charge_dict[q_j]:
                if 0<= k-q_j < N and 0 <= k+l-q_j < N and 0 <= k+l < N and k!=m:
                    denom = find_eigval(Hamiltonian, jump_operators, m, l, decay_rates) - find_eigval(Hamiltonian, jump_operators, k, l, decay_rates)
                    weight += gamma * J[k, k - q_j] * np.conj(J[k + l, k + l - q_j]) / denom

            if 0<= k-q_j < N:
                A[k, k - q_j] = weight

            weight = 0.0 + 0j
            for J, gamma in charge_dict[q_j]:
                if 0 <= k+q_j < N and 0 <= k+l+q_j < N and 0 <= k+l < N and k!=m:
                    denom = find_eigval(Hamiltonian, jump_operators, m, l, decay_rates) - find_eigval(Hamiltonian, jump_operators, k , l, decay_rates)
                    weight += np.conj(gamma * J[k + q_j,k] * np.conj(J[ k + l + q_j, k + l]) / denom)
            
            if 0 <= k+q_j < N:
                A[k, k + q_j] = weight

    return A 

if __name__ == "__main__":
    from read_params import read_params
    params = read_params("input_parameters.dat")
    print("Loaded parameters successfully.")
    A = construct_adjacency_matrix(params["hamil"], params["jump_ops"], m=0, l=0, decay_rates=params["decay_rates"])
    print("Adjacency matrix A for m=0, l=0:\n", A)