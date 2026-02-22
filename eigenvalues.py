import numpy as np
from preliminary_check import check_isotropicity_same_charge

def eigenvalues(Hamiltonian, jump_operators, decay_rates = None):

    if decay_rates is None:
        decay_rates = np.ones(len(jump_operators))

    result, charges = check_isotropicity_same_charge(Hamiltonian, jump_operators)
    
    eigvals = []
    for l in range(-Hamiltonian.shape[0]+1, Hamiltonian.shape[0]):
        for m in range(np.max([0, -l]), np.min([Hamiltonian.shape[0], Hamiltonian.shape[0]-l])):
            if 0<= m+l < Hamiltonian.shape[0]:
                eigval = -1j * (Hamiltonian[m, m] - Hamiltonian[m+l, m+l]) 
            for j in range(len(jump_operators)):
                q_j = charges[j]
                gamma_j = decay_rates[j]
                if 0 <= m+q_j < Hamiltonian.shape[0]:
                    eigval -= 0.5 * gamma_j * (np.abs(jump_operators[j][m+q_j, m])**2)
                if 0 <= m+l+q_j < Hamiltonian.shape[0] and 0 <= m+l < Hamiltonian.shape[0]:
                    eigval -= 0.5 * gamma_j * (np.abs(jump_operators[j][m+l + q_j, m+l])**2)
            eigvals.append(eigval)

    return np.sort(eigvals)

def find_eigval(Hamiltonian, jump_operators, m,l,decay_rates = None ):
    N = Hamiltonian.shape[0]
    if decay_rates is None:
        decay_rates = np.ones(len(jump_operators))


    eigval = 0.0 + 0j
    result, charges = check_isotropicity_same_charge(Hamiltonian, jump_operators)
    if 0<= m+l < Hamiltonian.shape[0]:
        eigval = -1j * (Hamiltonian[m, m] - Hamiltonian[m+l, m+l]) 
    for j in range(len(jump_operators)):
        q_j = charges[j]
        gamma_j = decay_rates[j]
        if 0 <= m+q_j < Hamiltonian.shape[0]:
            eigval -= 0.5 * gamma_j * (np.abs(jump_operators[j][m+q_j, m])**2)
        if 0 <= m+l+q_j < Hamiltonian.shape[0] and 0 <= m+l < Hamiltonian.shape[0]:
            eigval -= 0.5 * gamma_j * (np.abs(jump_operators[j][m+l + q_j, m+l])**2)
    return eigval


if __name__ == "__main__":
    from read_params import read_params
    params = read_params("input_parameters.dat")
    print("Loaded parameters successfully.")
    eigvals = eigenvalues(params["hamil"], params["jump_ops"], params["decay_rates"])
    print("Eigenvalues of the Lindbladian:", eigvals)