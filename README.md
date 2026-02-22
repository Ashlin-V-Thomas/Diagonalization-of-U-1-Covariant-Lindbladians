# Diagonalization of U(1) Covariant Lindbladians
Python implementation of diagonalization of U(1) covariant single mode lindbladians using the exact solutions obtained by leveranging the block diagonal structure of the Liouvillian. The implementation applies to models where all jump operators carry the same U(1) charge.

## U(1) Covariant Lindbladians
A Lindbladian is said to be U(1) covariant if it commutes with a U(1) generator $\hat{N}$, meaning the evolution preserves the eigenspaces of $\hat{N}$. The 
GKSL equation takes the form - 

$$\frac{d\rho}{dt} = \mathcal{L}[\rho] = -i[H, \rho] + \sum_k \gamma_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)$$

where $H$ is the Hamiltonian, $L_k$ are the jump operators, and $\gamma_k$ are the decay rates.
In this work, we focus on single mode oscillator and spin systems.

For U(1) covariance, we require:
- $[H, \hat{N}] = 0$
- $[\hat{N}, L_k] = q_k L_k$ for some charge $q_k \in \mathbb{Z}$

In addition, we assume all jump operators carry charges of the same sign, i.e., $sgn(q_k) = sgn(q_{k'})$ for all $k, k'$.

## Diagonalization Procedure
The diagonalization of the Lindbladian can be achieved by leveraging the block diagonal structure of the Liouvillian in the eigenbasis of $\hat{\mathcal{N}}(\cdot) = [\hat{N}, \cdot]$. Using this symmetry, we recast the problem of diagonalization into a path counting problem on a weighted directed graph - you can find the details of the procedure in the notes directory.
This approach yields exact solutions for the eigenvalues and eigenvectors of the Lindbladian, which are implemented in this   repository.

## Usage
To use the implementation, edit the 'input_parameters.dat' file to specify the details of your model. Then, run the 'diagonalize.py' script to obtain the eigenvalues and eigenvectors of the Lindbladian. The results will be saved in the 'output' directory.
