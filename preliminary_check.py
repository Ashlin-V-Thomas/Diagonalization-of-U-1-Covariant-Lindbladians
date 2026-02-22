import numpy as np

def extract_charges(jump_operators,tol =1e-15):
    charges = []

    for idx, J in enumerate(jump_operators):
        # Find all non-zero (above tolerance) entries
        rows, cols = np.where(np.abs(J) > tol)

        if len(rows) == 0:
            raise ValueError(f"Jump operator #{idx} is zero (no non-zero elements).")

        # Compute diagonal offsets: q_j = col - row
        diag_offsets = cols - rows
        unique_offsets = np.unique(diag_offsets)

        if len(unique_offsets) != 1:
            raise ValueError(
                f"Jump operator #{idx} has elements on multiple diagonals: {unique_offsets}. "
                "Hence it does not have a well-defined U(1) charge."
            )

        q_j = int(unique_offsets[0])
        charges.append(q_j)

    return -1*np.array(charges)


def check_isotropicity_same_charge(Hamiltonian, jump_operators, tol=1e-15):

    N = np.diag(np.arange(Hamiltonian.shape[0]))
    if np.linalg.norm(Hamiltonian @ N - N @ Hamiltonian) > tol:
        raise ValueError("Hamiltonian does not commute with the number operator, hence the Lindbladian is not U(1) covariant.")
        return False, None

    charges = extract_charges(jump_operators, tol=tol)
    s = np.sign(charges[0])


    if not all(np.sign(q) == s for q in charges):
        raise ValueError(f"Not all jump operators carry U(1) charges of the same sign, which violates the condition required for the diagonalization procedure. Charges: {charges}")
        return False, None

    return True, charges


if __name__ == "__main__":
    from read_params import read_params
    params = read_params("input_parameters.dat")
    print("Loaded parameters successfully.")
    is_valid, charges = check_isotropicity_same_charge(params["hamil"], params["jump_ops"])
    if is_valid:
        print("The model is valid for diagonalization. Extracted charges:", charges)
    else:
        print("The model is not valid for diagonalization.")