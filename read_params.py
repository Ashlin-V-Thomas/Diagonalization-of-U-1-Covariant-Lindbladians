import numpy as np
import re
import ast


def _remove_comments(line):
    return line.split("#")[0].strip()


def _parse_bool(val):
    if val not in ["True", "False"]:
        raise ValueError(f"Invalid boolean value: {val}")
    return val == "True"


# ---------- Operator builders ----------

def create_annihilation(N):
    a = np.zeros((N, N), dtype=np.complex128)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a


def create_creation(N):
    return create_annihilation(N).conj().T


def create_number(N):
    return np.diag(np.arange(N))


# ---------- Parsing helpers ----------

def _parse_hamiltonian(val, N):
    try:
        coeffs = np.array(ast.literal_eval(val), dtype=float)
    except Exception:
        raise ValueError("Hamiltonian must be a list of numbers")

    n_op = create_number(N)
    H = np.zeros((N, N), dtype=np.complex128)

    for k, c in enumerate(coeffs):
        if k == 0:
            H += c * np.eye(N)
        else:
            H += c * np.linalg.matrix_power(n_op, k)

    return H

def _parse_jump_operators(jump_val, rate_val, N):
    try:
        jump_list = ast.literal_eval(jump_val)
    except Exception:
        raise ValueError("Jump_operators must be a list of [m, n] pairs")

    # Parse decay rates
    if rate_val is None:
        decay_rates = [1.0] * len(jump_list)
    else:
        try:
            decay_rates = list(ast.literal_eval(rate_val))
        except Exception:
            raise ValueError("decay_rates must be a list of numbers")

        if len(decay_rates) != len(jump_list):
            raise ValueError("Length of decay_rates must match Jump_operators")

    a = create_annihilation(N)
    adag = create_creation(N)

    jump_ops = []

    for idx, pair in enumerate(jump_list):
        if not (isinstance(pair, list) or isinstance(pair, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid jump operator at index {idx}: {pair}")

        m, n = pair

        if not (isinstance(m, int) and isinstance(n, int) and m >= 0 and n >= 0):
            raise ValueError(f"Invalid powers (m, n) at index {idx}: {pair}")

        # Construct operator WITHOUT gamma
        L = np.linalg.matrix_power(adag, m) @ np.linalg.matrix_power(a, n)
        jump_ops.append(L)

    return jump_ops, np.array(decay_rates, dtype=float)


# ---------- Main ----------

def read_params(filename="input_parameters.dat"):
    raw = {}

    with open(filename, "r") as f:
        for line in f:
            line = _remove_comments(line)
            if not line or "=" not in line:
                continue

            key, val = map(str.strip, line.split("=", 1))
            raw[key] = val

    # ---------- REQUIRED KEYS ----------
    required_keys = [
        "Hilbert_space_dimension",
        "Hamiltonian",
        "jump_operators",
        "decay_rates",
        "save_lindbladian",
        "save_eigenvalues",
        "save_eigenvectors",
        "save_adjacency_matrix",
        "plot_spectrum",
    ]

    missing = [k for k in required_keys if k not in raw]
    if missing:
        raise KeyError(f"Missing required parameters: {missing}")

    # ---------- PARSING ----------
    try:
        N = int(raw["Hilbert_space_dimension"])
    except Exception:
        raise ValueError("Hilbert_space_dimension must be an integer")

    if N <= 0:
        raise ValueError("Hilbert_space_dimension must be positive")

    params = {}
    params["N"] = N

    params["hamil"] = _parse_hamiltonian(raw["Hamiltonian"], N)
    params["jump_ops"], params["decay_rates"] = _parse_jump_operators(raw["jump_operators"], raw["decay_rates"], N)

    params["save_lindbladian"] = _parse_bool(raw["save_lindbladian"])
    params["save_eigenvalues"] = _parse_bool(raw["save_eigenvalues"])
    params["save_eigenvectors"] = _parse_bool(raw["save_eigenvectors"])
    params["plot_spectrum"] = _parse_bool(raw["plot_spectrum"])
    params["save_adjacency_matrix"] = _parse_bool(raw["save_adjacency_matrix"])
    params["adj_mat_mode_to_save"] = list(ast.literal_eval(raw["adj_mat_mode_to_save"]))
    params["clean_outputs"] = _parse_bool(raw.get("clean_outputs", "False"))
    return params


if __name__ == "__main__":
    params = read_params("input_parameters.dat")

    print("N =", params["N"])
    print("Hamiltonian shape:", params["hamil"].shape)
    print("Number of jump operators:", len(params["jump_ops"]))
    print("Save Lindbladian:", params["save_lindbladian"])
    print("Save eigenvalues:", params["save_eigenvalues"])
    print("Save eigenvectors:", params["save_eigenvectors"])
    print("Adjacency matrix modes to save:", params["adj_mat_mode_to_save"])
