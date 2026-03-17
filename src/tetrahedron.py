"""
Regular tetrahedron geometry: vertices on the unit sphere, symmetry group,
isotropy tensor, and directional derivatives.

Vertex convention (inscribed in unit sphere, centroid at origin):
  e_0 = (0, 0, 1)
  e_1 = (2√2/3, 0, -1/3)
  e_2 = (-√2/3, √6/3, -1/3)
  e_3 = (-√2/3, -√6/3, -1/3)
"""

import sympy as sp
from sympy import Matrix, sqrt, Rational, eye, zeros, simplify

# ---------------------------------------------------------------------------
# Vertex coordinates (sympy exact)
# ---------------------------------------------------------------------------
e0 = Matrix([0, 0, 1])
e1 = Matrix([2 * sqrt(2) / 3, 0, Rational(-1, 3)])
e2 = Matrix([-sqrt(2) / 3, sqrt(6) / 3, Rational(-1, 3)])
e3 = Matrix([-sqrt(2) / 3, -sqrt(6) / 3, Rational(-1, 3)])

vertices = [e0, e1, e2, e3]

# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

def verify_unit_vectors():
    """Check |e_a| = 1 for all a."""
    return {f'|e_{a}|=1': simplify(v.dot(v) - 1) == 0
            for a, v in enumerate(vertices)}


def verify_centroid():
    """Check Σ_a e_a = 0 (centroid at origin)."""
    total = sum(vertices, Matrix([0, 0, 0]))
    return simplify(total) == Matrix([0, 0, 0])


def verify_dot_products():
    """Check e_a · e_b = -1/3 for a ≠ b."""
    results = {}
    for a in range(4):
        for b in range(a + 1, 4):
            dot = simplify(vertices[a].dot(vertices[b]))
            results[f'e_{a}·e_{b}'] = dot == Rational(-1, 3)
    return results


def isotropy_tensor():
    """
    Compute Σ_a e_a^i e_a^j and verify it equals (4/3) δ_{ij}.

    Returns the 3x3 matrix Σ_a e_a e_a^T.
    """
    T = zeros(3)
    for v in vertices:
        T += v * v.T
    return T.applyfunc(simplify)


def verify_isotropy():
    """Check Σ_a e_a^i e_a^j = (4/3) δ_{ij}."""
    T = isotropy_tensor()
    expected = Rational(4, 3) * eye(3)
    return (T - expected).applyfunc(simplify) == zeros(3)


# ---------------------------------------------------------------------------
# Numpy versions for numerical work
# ---------------------------------------------------------------------------

def vertices_numpy():
    """Return vertex coordinates as a list of numpy arrays."""
    import numpy as np
    return [np.array(v.tolist(), dtype=float).flatten() for v in vertices]
