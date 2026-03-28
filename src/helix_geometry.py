"""
Exact BC helix geometry from the analytic vertex formula.

The Boerdijk-Coxeter helix has vertices at:

    v_k = (r cos(kθ), r sin(kθ), k h)

where r = 3√3/10, h = 1/√10, θ = arccos(-2/3), and the edge length is 1.

Tetrahedron n has vertices {v_n, v_{n+1}, v_{n+2}, v_{n+3}}.
Its centroid is c_n = (v_n + v_{n+1} + v_{n+2} + v_{n+3}) / 4.

To match the walk code's convention (tetrahedra inscribed in the unit sphere,
centroid at origin, R-helix pattern [1,3,0,2]), we apply a Procrustes
alignment (rotation + scale + translation) fitted to the first 100 centroids.

All geometry is computed from the closed-form vertex formula — no sequential
reflections, no reorthogonalization, no accumulated numerical error.
"""

import numpy as np

# BC helix parameters (unit edge length)
THETA_BC = np.arccos(-2 / 3)  # twist per vertex ≈ 131.81°
R_VERTEX = 3 * np.sqrt(3) / 10  # cylinder radius
H_VERTEX = 1 / np.sqrt(10)  # axial pitch per vertex

# Dirac matrices
_ALPHA = [
    np.array([[0, 0, 0, 1], [0, 0, 1, 0],
              [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex),
    np.array([[0, 0, 0, -1j], [0, 0, 1j, 0],
              [0, -1j, 0, 0], [1j, 0, 0, 0]], dtype=complex),
    np.array([[0, 0, 1, 0], [0, 0, 0, -1],
              [1, 0, 0, 0], [0, -1, 0, 0]], dtype=complex),
]
_NU = np.sqrt(7) / 4


def _formula_vertex(k):
    """Raw BC helix vertex position (unit edge, formula frame)."""
    k = np.asarray(k, dtype=float)
    return np.column_stack([
        R_VERTEX * np.cos(k * THETA_BC),
        R_VERTEX * np.sin(k * THETA_BC),
        k * H_VERTEX,
    ]) if k.ndim > 0 else np.array([
        R_VERTEX * np.cos(k * THETA_BC),
        R_VERTEX * np.sin(k * THETA_BC),
        k * H_VERTEX,
    ])


def _formula_centroid(n):
    """Centroid of n-th tetrahedron in formula frame."""
    ks = np.array([n, n + 1, n + 2, n + 3], dtype=float)
    return _formula_vertex(ks).mean(axis=0)


def _compute_alignment(N_fit=100):
    """
    Compute Procrustes alignment (rotation + scale + translation) from
    the formula frame to the walk code frame, using N_fit centroid positions.

    The walk code uses:
        - Tetrahedra inscribed in the unit sphere (edge = 2√6/3)
        - R-helix pattern [1, 3, 0, 2]
        - Initial vertex directions: (0,0,1), (2√2/3,0,-1/3), etc.

    Returns (R, scale, t) such that:
        walk_pos = scale * R @ formula_pos + t
    """
    # Generate walk-code positions via sequential reflection
    dirs = np.array([
        [0, 0, 1],
        [2 * np.sqrt(2) / 3, 0, -1 / 3],
        [-np.sqrt(2) / 3, np.sqrt(6) / 3, -1 / 3],
        [-np.sqrt(2) / 3, -np.sqrt(6) / 3, -1 / 3],
    ])
    pat = [1, 3, 0, 2]
    pos_walk = [np.zeros(3)]
    for n in range(N_fit):
        face = pat[n % 4]
        e = dirs[face].copy()
        pos_walk.append(pos_walk[-1] + e * (-2 / 3))
        for a in range(4):
            dirs[a] = dirs[a] - 2 * np.dot(dirs[a], e) * e
        if (n + 1) % 4 == 0:
            m = dirs.mean(axis=0)
            dirs -= m
            for a in range(4):
                nm = np.linalg.norm(dirs[a])
                if nm > 1e-15:
                    dirs[a] /= nm
    pos_walk = np.array(pos_walk[:N_fit + 1])

    # Generate formula centroids
    pos_formula = np.array([_formula_centroid(n) for n in range(N_fit + 1)])

    # Procrustes: find R, scale, t minimizing ||scale * R @ F + t - W||
    W_mean = pos_walk.mean(axis=0)
    F_mean = pos_formula.mean(axis=0)
    W_c = pos_walk - W_mean
    F_c = pos_formula - F_mean

    scale = np.sqrt(np.sum(W_c ** 2) / np.sum(F_c ** 2))
    F_s = F_c * scale

    H = F_s.T @ W_c
    U, S, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, det]) @ U.T

    t = W_mean - scale * R @ F_mean

    return R, scale, t


# Precompute alignment at import time
_R_ALIGN, _SCALE, _T_ALIGN = _compute_alignment()


def vertex(k):
    """
    Position of the k-th BC helix vertex in the walk code's frame.

    Parameters
    ----------
    k : int or array of int

    Returns
    -------
    pos : ndarray, shape (3,) or (len(k), 3)
    """
    v_f = _formula_vertex(k)
    if v_f.ndim == 1:
        return _SCALE * _R_ALIGN @ v_f + _T_ALIGN
    return _SCALE * (v_f @ _R_ALIGN.T) + _T_ALIGN


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
    tau = np.diag([_NU, _NU, -_NU, -_NU]).astype(complex)
    for a in range(3):
        tau += 0.75 * d[a] * _ALPHA[a]
    return tau


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
