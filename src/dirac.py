"""
Dirac equation infrastructure: gamma matrices, alpha matrices, Clifford algebra.

Convention: Dirac (standard) representation.
  gamma^0 = diag(I_2, -I_2)
  gamma^i = [[0, sigma_i], [-sigma_i, 0]]
  alpha^i = gamma^0 @ gamma^i = [[0, sigma_i], [sigma_i, 0]]
  beta    = gamma^0

Metric signature: (+, -, -, -)
"""

import sympy as sp
from sympy import Matrix, eye, zeros, I, sqrt, Rational

# ---------------------------------------------------------------------------
# Pauli matrices (2x2, sympy)
# ---------------------------------------------------------------------------
sigma_x = Matrix([[0, 1], [1, 0]])
sigma_y = Matrix([[0, -I], [I, 0]])
sigma_z = Matrix([[1, 0], [0, -1]])

pauli = [sigma_x, sigma_y, sigma_z]

# ---------------------------------------------------------------------------
# Identity matrices
# ---------------------------------------------------------------------------
I2 = eye(2)
I4 = eye(4)

# ---------------------------------------------------------------------------
# Gamma matrices in Dirac representation (4x4, sympy)
# ---------------------------------------------------------------------------

def _block(a, b, c, d):
    """Build a 4x4 matrix from four 2x2 blocks: [[a, b], [c, d]]."""
    return a.row_join(b).col_join(c.row_join(d))


gamma_0 = _block(I2, zeros(2), zeros(2), -I2)

gamma_1 = _block(zeros(2), sigma_x, -sigma_x, zeros(2))
gamma_2 = _block(zeros(2), sigma_y, -sigma_y, zeros(2))
gamma_3 = _block(zeros(2), sigma_z, -sigma_z, zeros(2))

gamma = [gamma_0, gamma_1, gamma_2, gamma_3]

# ---------------------------------------------------------------------------
# Alpha matrices: alpha^i = gamma^0 @ gamma^i
# ---------------------------------------------------------------------------
alpha_1 = (gamma_0 * gamma_1).applyfunc(sp.simplify)
alpha_2 = (gamma_0 * gamma_2).applyfunc(sp.simplify)
alpha_3 = (gamma_0 * gamma_3).applyfunc(sp.simplify)

alpha = [alpha_1, alpha_2, alpha_3]

# ---------------------------------------------------------------------------
# Beta matrix
# ---------------------------------------------------------------------------
beta = gamma_0

# ---------------------------------------------------------------------------
# gamma_5 = i * gamma^0 * gamma^1 * gamma^2 * gamma^3
# ---------------------------------------------------------------------------
gamma_5 = (I * gamma_0 * gamma_1 * gamma_2 * gamma_3).applyfunc(sp.simplify)

# ---------------------------------------------------------------------------
# Minkowski metric eta^{mu,nu} = diag(+1, -1, -1, -1)
# ---------------------------------------------------------------------------
eta = sp.diag(1, -1, -1, -1)

# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

def verify_clifford_algebra():
    """Check {gamma^mu, gamma^nu} = 2 eta^{mu,nu} I_4 for all mu, nu."""
    results = {}
    for mu in range(4):
        for nu in range(4):
            anticomm = (gamma[mu] * gamma[nu] + gamma[nu] * gamma[mu]).applyfunc(sp.simplify)
            expected = 2 * eta[mu, nu] * I4
            ok = (anticomm - expected).applyfunc(sp.simplify) == zeros(4)
            results[(mu, nu)] = ok
    return results


def verify_alpha_properties():
    """Check that alpha matrices are Hermitian, square to I, and anticommute."""
    results = {}
    for i in range(3):
        # Hermitian
        results[f'alpha_{i+1}_hermitian'] = (
            alpha[i] - alpha[i].adjoint()
        ).applyfunc(sp.simplify) == zeros(4)
        # Squares to identity
        results[f'alpha_{i+1}_squared'] = (
            alpha[i] * alpha[i] - I4
        ).applyfunc(sp.simplify) == zeros(4)

    # Anticommutation: {alpha_i, alpha_j} = 2 delta_{ij} I
    for i in range(3):
        for j in range(i + 1, 3):
            anticomm = (alpha[i] * alpha[j] + alpha[j] * alpha[i]).applyfunc(sp.simplify)
            results[f'alpha_{i+1}_alpha_{j+1}_anticomm'] = anticomm == zeros(4)

    # {alpha_i, beta} = 0
    for i in range(3):
        anticomm = (alpha[i] * beta + beta * alpha[i]).applyfunc(sp.simplify)
        results[f'alpha_{i+1}_beta_anticomm'] = anticomm == zeros(4)

    return results


def spinor_rotation_generator(mu, nu):
    """Spinor rotation generator S_{mu,nu} = (i/4)[gamma^mu, gamma^nu]."""
    comm = gamma[mu] * gamma[nu] - gamma[nu] * gamma[mu]
    return (I * Rational(1, 4) * comm).applyfunc(sp.simplify)
