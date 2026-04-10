"""
Exact BC helix geometry from the analytic vertex formula.

The Boerdijk-Coxeter helix has vertices at:

    v_k = (r cos(kθ), r sin(kθ), k h)

where r = 3√3/10, h = 1/√10, θ = arccos(-2/3), and the edge length is 1.

Tetrahedron n has vertices {v_n, v_{n+1}, v_{n+2}, v_{n+3}}.
Its centroid is c_n = (v_n + v_{n+1} + v_{n+2} + v_{n+3}) / 4.

All geometry is computed from the closed-form vertex formula — no sequential
reflections, no reorthogonalization, no accumulated numerical error.
"""

import numpy as np
from src.dirac import alpha as ALPHA, beta as BETA
from src.tau_operators import NU

# BC helix parameters (unit edge length)
THETA_BC = np.arccos(-2 / 3)  # twist per vertex ≈ 131.81°
R_VERTEX = 3 * np.sqrt(3) / 10  # cylinder radius
H_VERTEX = 1 / np.sqrt(10)  # axial pitch per vertex


def vertex(k):
    """
    Position of the k-th BC helix vertex.

    Parameters
    ----------
    k : int or array of int

    Returns
    -------
    pos : ndarray, shape (3,) or (len(k), 3)
    """
    k = np.asarray(k, dtype=float)
    if k.ndim > 0:
        return np.column_stack([
            R_VERTEX * np.cos(k * THETA_BC),
            R_VERTEX * np.sin(k * THETA_BC),
            k * H_VERTEX,
        ])
    return np.array([
        R_VERTEX * np.cos(k * THETA_BC),
        R_VERTEX * np.sin(k * THETA_BC),
        k * H_VERTEX,
    ])


def centroid(n):
    """
    Centroid of the n-th tetrahedron in the walk code's frame.

    Parameters
    ----------
    n : int or array of int

    Returns
    -------
    pos : ndarray, shape (3,) or (len(n), 3)
    """
    n = np.asarray(n)
    scalar = n.ndim == 0
    n = np.atleast_1d(n)

    pos = np.zeros((len(n), 3))
    for k in range(4):
        pos += vertex(n + k)
    pos /= 4

    return pos[0] if scalar else pos


def vertex_directions(n):
    """
    The 4 vertex direction vectors from the centroid of tetrahedron n,
    in the walk code's frame. These are unit vectors on the unit sphere.

    Parameters
    ----------
    n : int

    Returns
    -------
    dirs : ndarray, shape (4, 3)
    """
    verts = np.array([vertex(n + k) for k in range(4)])
    c = verts.mean(axis=0)
    dirs = verts - c
    for i in range(4):
        dirs[i] /= np.linalg.norm(dirs[i])
    return dirs


def exit_direction(n, pattern=None):
    """
    The exit direction at site n.

    The exit direction is the vertex direction for face pattern[n % 4].
    However, the vertex ordering in the sliding window changes at each step
    because different faces are exited. We determine the correct exit
    direction by finding which vertex of tetrahedron n is NOT shared
    with tetrahedron n+1.

    For the step from tetrahedron n to n+1:
        tet n   has vertices {v_n,   v_{n+1}, v_{n+2}, v_{n+3}}
        tet n+1 has vertices {v_{n+1}, v_{n+2}, v_{n+3}, v_{n+4}}
    The dropped vertex is v_n. Its direction from the centroid gives
    the exit direction (pointing from centroid TOWARD the dropped vertex).

    Parameters
    ----------
    n : int
    pattern : ignored (kept for API compatibility)

    Returns
    -------
    d : ndarray, shape (3,)
        Unit exit direction.
    """
    c = centroid(n)
    v_dropped = vertex(n)  # v_n is the vertex not shared with tet n+1
    d = v_dropped - c
    return d / np.linalg.norm(d)


def make_tau(d):
    """
    Construct the τ operator from a unit direction vector.

    τ = (√7/4) β + (3/4)(d · α)
    """
    tau = NU * BETA + 0.75 * sum(d[a] * ALPHA[a] for a in range(3))
    return tau


def entry_direction(n):
    """
    The entry direction at site n.

    Points from the centroid toward vertex n+3 — the face through which
    the walker entered when advancing along the chain.
    """
    c = centroid(n)
    v_entry = vertex(n + 3)
    d = v_entry - c
    return d / np.linalg.norm(d)


def build_entry_taus(N):
    """Compute τ operators from entry directions for N sites."""
    taus = np.zeros((N, 4, 4), dtype=complex)
    for n in range(N):
        d = entry_direction(n)
        taus[n] = make_tau(d)
    return taus


def build_taus(N, pattern=None):
    """
    Compute τ operators at each site along a BC helix of N sites.

    Uses the analytic vertex formula — no sequential reflections.

    Parameters
    ----------
    N : int
        Number of sites.
    pattern : ignored (kept for API compatibility)

    Returns
    -------
    taus : ndarray, shape (N, 4, 4)
        τ operator at each site.
    """
    taus = np.zeros((N, 4, 4), dtype=complex)
    for n in range(N):
        d = exit_direction(n)
        taus[n] = make_tau(d)
    return taus
