import numpy as np

def construct_lindbladian(H, jump_operators):
    n = H.shape[0]
    I = np.eye(n, dtype=complex)

    # Hamiltonian part: -i (I ⊗ H - Hᵀ ⊗ I)
    LH = -1j * (np.kron(I, H) - np.kron(H.T, I))

    LL = np.zeros((n*n, n*n), dtype=complex)

    for L in jump_operators:
        K = L.conj().T @ L

        term1 = np.kron(L.conj(), L)          # L ρ L†
        term2 = -0.5 * np.kron(I, K)          # -1/2 (L† L ρ)
        term3 = -0.5 * np.kron(K.T, I)        # -1/2 (ρ L† L)

        LL += term1 + term2 + term3

    return LH + LL