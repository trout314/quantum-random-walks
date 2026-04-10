"""
Dirac equation infrastructure: gamma matrices, alpha matrices, Clifford algebra.

Convention: Dirac (standard) representation.
  gamma^0 = diag(I_2, -I_2)
  gamma^i = [[0, sigma_i], [-sigma_i, 0]]
  alpha^i = gamma^0 @ gamma^i = [[0, sigma_i], [sigma_i, 0]]
  beta    = gamma^0

Metric signature: (+, -, -, -)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Numpy Dirac matrices (primary, used by all numerical code)
# ---------------------------------------------------------------------------
alpha = [
    np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=complex),
    np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]], dtype=complex),
    np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]], dtype=complex),
]
beta = np.diag([1, 1, -1, -1]).astype(complex)
I4 = np.eye(4, dtype=complex)

# gamma^0 = beta, gamma^i = beta @ alpha^i
gamma = [beta] + [beta @ alpha[i] for i in range(3)]
gamma_5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]


# ---------------------------------------------------------------------------
# Sympy Dirac matrices (for symbolic algebra in tests and derivations)
# ---------------------------------------------------------------------------
import sympy as sp
from sympy import Matrix, eye, zeros, I, sqrt, Rational

_I2 = eye(2)
_I4_sym = eye(4)

_sigma = [
    Matrix([[0, 1], [1, 0]]),
    Matrix([[0, -I], [I, 0]]),
    Matrix([[1, 0], [0, -1]]),
]

def _block(a, b, c, d):
    """Build a 4x4 matrix from four 2x2 blocks: [[a, b], [c, d]]."""
    return a.row_join(b).col_join(c.row_join(d))

gamma_sym = [
    _block(_I2, zeros(2), zeros(2), -_I2),
    _block(zeros(2), _sigma[0], -_sigma[0], zeros(2)),
    _block(zeros(2), _sigma[1], -_sigma[1], zeros(2)),
    _block(zeros(2), _sigma[2], -_sigma[2], zeros(2)),
]
alpha_sym = [
    (gamma_sym[0] * gamma_sym[i+1]).applyfunc(sp.simplify) for i in range(3)
]
beta_sym = gamma_sym[0]
gamma_5_sym = (I * gamma_sym[0] * gamma_sym[1] * gamma_sym[2] * gamma_sym[3]).applyfunc(sp.simplify)

# Minkowski metric
eta = sp.diag(1, -1, -1, -1)

# ---------------------------------------------------------------------------
# Verification functions (symbolic)
# ---------------------------------------------------------------------------

def verify_clifford_algebra():
    """Check {gamma^mu, gamma^nu} = 2 eta^{mu,nu} I_4 for all mu, nu."""
    results = {}
    for mu in range(4):
        for nu in range(4):
            anticomm = (gamma_sym[mu] * gamma_sym[nu] + gamma_sym[nu] * gamma_sym[mu]).applyfunc(sp.simplify)
            expected = 2 * eta[mu, nu] * _I4_sym
            ok = (anticomm - expected).applyfunc(sp.simplify) == zeros(4)
            results[(mu, nu)] = ok
    return results


def verify_alpha_properties():
    """Check that alpha matrices are Hermitian, square to I, and anticommute."""
    results = {}
    for i in range(3):
        results[f'alpha_{i+1}_hermitian'] = (
            alpha_sym[i] - alpha_sym[i].adjoint()
        ).applyfunc(sp.simplify) == zeros(4)
        results[f'alpha_{i+1}_squared'] = (
            alpha_sym[i] * alpha_sym[i] - _I4_sym
        ).applyfunc(sp.simplify) == zeros(4)

    for i in range(3):
        for j in range(i + 1, 3):
            anticomm = (alpha_sym[i] * alpha_sym[j] + alpha_sym[j] * alpha_sym[i]).applyfunc(sp.simplify)
            results[f'alpha_{i+1}_alpha_{j+1}_anticomm'] = anticomm == zeros(4)

    for i in range(3):
        anticomm = (alpha_sym[i] * beta_sym + beta_sym * alpha_sym[i]).applyfunc(sp.simplify)
        results[f'alpha_{i+1}_beta_anticomm'] = anticomm == zeros(4)

    return results


def spinor_rotation_generator(mu, nu):
    """Spinor rotation generator S_{mu,nu} = (i/4)[gamma^mu, gamma^nu]."""
    comm = gamma_sym[mu] * gamma_sym[nu] - gamma_sym[nu] * gamma_sym[mu]
    return (I * Rational(1, 4) * comm).applyfunc(sp.simplify)
