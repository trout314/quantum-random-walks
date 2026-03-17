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
# Reflections and walker paths
# ---------------------------------------------------------------------------

def reflect_directions(dirs, step_index):
    """
    Reflect a tetrahedron through the plane perpendicular to dirs[step_index].

    When a walker at a site with directions `dirs` steps in direction
    dirs[step_index], the directions at the new site are the reflection
    of all four directions through the plane perpendicular to the step direction:

        R_a(v) = v - 2(v · e_a) e_a

    Parameters
    ----------
    dirs : list of 4 sympy Matrix (3×1)
        The current tetrahedral directions.
    step_index : int (0-3)
        Which direction the walker steps in.

    Returns
    -------
    list of 4 sympy Matrix (3×1)
        The tetrahedral directions at the new site.
    """
    e = dirs[step_index]
    return [(d - 2 * d.dot(e) * e).applyfunc(simplify) for d in dirs]


def directions_after_path(path):
    """
    Compute the tetrahedral directions at the end of a walker path.

    Starting from the initial tetrahedron {e_0, e_1, e_2, e_3}, applies
    the sequence of reflections determined by the path. Each element of
    `path` is an index (0-3) indicating which of the current four
    directions the walker steps in.

    Parameters
    ----------
    path : list of int
        Sequence of direction indices (each 0-3).

    Returns
    -------
    list of 4 sympy Matrix (3×1)
        The tetrahedral directions at the endpoint.
    """
    dirs = list(vertices)
    for step_index in path:
        dirs = reflect_directions(dirs, step_index)
    return dirs


def position_after_path(path):
    """
    Compute the walker's position at the end of a path.

    Starting from the origin with directions {e_0, e_1, e_2, e_3},
    the walker takes steps along the indicated directions. The position
    is the sum of all step vectors.

    Parameters
    ----------
    path : list of int
        Sequence of direction indices (each 0-3).

    Returns
    -------
    pos : sympy Matrix (3×1)
        The walker's position.
    dirs : list of 4 sympy Matrix (3×1)
        The tetrahedral directions at the endpoint.
    """
    dirs = list(vertices)
    pos = Matrix([0, 0, 0])
    for step_index in path:
        pos = (pos + dirs[step_index]).applyfunc(simplify)
        dirs = reflect_directions(dirs, step_index)
    return pos, dirs


# ---------------------------------------------------------------------------
# Numpy versions for numerical work
# ---------------------------------------------------------------------------

def vertices_numpy():
    """Return vertex coordinates as a list of numpy arrays."""
    import numpy as np
    return [np.array(v.tolist(), dtype=float).flatten() for v in vertices]
