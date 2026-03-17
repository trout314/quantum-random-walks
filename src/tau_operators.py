"""
τ operator construction for the tetrahedral quantum walk.

We need 4 operators τ_0, τ_1, τ_2, τ_3 (4×4 matrices) satisfying:
  (1) Dirac correspondence: Σ_a e_a^i τ_a = α_i  for i = 1, 2, 3
  (2) Involutory: τ_a² = I  (eigenvalues ±1)
  (3) Hermitian: τ_a = τ_a†
  (4) Spectrum: eigenvalues {-1, -1, 1, 1}  (unitarily equivalent to diag(-1,-1,1,1))

Derivation:
  Ansatz: τ_a = ν β + Σ_j s_a^j α_j  where β, α_j all anticommute pairwise.

  Since {β, α_i} = 0 and {α_i, α_j} = 2δ_{ij}I, cross terms vanish:
    τ_a² = (ν² + |s_a|²) I

  So τ_a² = I requires ν² + |s_a|² = 1.

  Dirac correspondence: Σ_a e_a^i τ_a = α_i
    => Σ_a e_a^i ν β + Σ_a e_a^i s_a^j α_j = 0·β + δ_{ij} α_j
    => ν (Σ_a e_a^i) = 0  (automatic since Σ_a e_a = 0)
    => Σ_a e_a^i s_a^j = δ_{ij}

  The constraint Σ_a e_a^i s_a^j = δ_{ij} means E @ S = I₃ where E is 3×4
  (columns = e_a) and S is 4×3 (rows = s_a). Using isotropy (E E^T = (4/3)I₃):
    Particular solution: S = (3/4) E^T, i.e., s_a = (3/4) e_a
    General solution: s_a = (3/4) e_a + μ for any constant 3-vector μ

  Unit norm: ν² + |s_a|² = 1  must hold for all a.
    ν² + (9/16)|e_a|² + (3/2)(e_a · μ) + |μ|² = 1
    Since |e_a| = 1: ν² + 9/16 + |μ|² + (3/2)(e_a · μ) = 1

  This must be independent of a, so e_a · μ = const for all a.
  But Σ_a e_a = 0 forces this constant to be 0, and checking vertex by vertex
  shows μ = 0 is the only solution.

  Therefore: s_a = (3/4) e_a  and  ν² = 1 - 9/16 = 7/16, giving ν = ±√7/4.

Result:
  τ_a = ±(√7/4) β + (3/4)(e_a · α)
"""

import sympy as sp
from sympy import Matrix, Rational, sqrt, simplify, eye, zeros, I

from src.dirac import alpha, beta, gamma_5, I4
from src.tetrahedron import vertices


def construct_tau(sign=1):
    """
    Construct τ_a = sign*(√7/4) β + (3/4)(e_a · α).

    Parameters
    ----------
    sign : +1 or -1
        Sign of the β coefficient.

    Returns
    -------
    list of 4 sympy Matrix (4×4)
    """
    nu = sign * sqrt(7) / 4
    taus = []
    for a in range(4):
        e = vertices[a]
        tau_a = nu * beta + Rational(3, 4) * sum(
            (e[i] * alpha[i] for i in range(3)), zeros(4)
        )
        taus.append(tau_a.applyfunc(simplify))
    return taus


def verify_dirac_correspondence(taus):
    """Check Σ_a e_a^i τ_a = α_i for i = 1, 2, 3."""
    results = {}
    for i in range(3):
        lhs = sum(
            (vertices[a][i] * taus[a] for a in range(4)), zeros(4)
        ).applyfunc(simplify)
        diff = (lhs - alpha[i]).applyfunc(simplify)
        results[f'direction_{i+1}'] = diff == zeros(4)
    return results


def verify_involutory(taus):
    """Check τ_a² = I for all a."""
    return {f'tau_{a}^2=I': (
        taus[a] * taus[a] - I4
    ).applyfunc(simplify) == zeros(4) for a in range(4)}


def verify_hermitian(taus):
    """Check each τ_a is Hermitian."""
    return {f'tau_{a}_hermitian': (
        taus[a] - taus[a].adjoint()
    ).applyfunc(simplify) == zeros(4) for a in range(4)}


def verify_eigenvalues(taus):
    """Return eigenvalues of each τ_a (expect {-1: 2, 1: 2})."""
    return {f'tau_{a}': taus[a].eigenvals() for a in range(4)}


def verify_all(taus):
    """Run all verification checks. Returns dict of results."""
    results = {}
    results['dirac_correspondence'] = verify_dirac_correspondence(taus)
    results['involutory'] = verify_involutory(taus)
    results['hermitian'] = verify_hermitian(taus)
    results['eigenvalues'] = verify_eigenvalues(taus)
    return results
