"""
Quantum walk operators on the BC helix.

Constructs the shift operator S_R along a right-handed BC helix,
with frame transport to ensure unitarity.
"""

import numpy as np

# Dirac matrices (numerical, Dirac representation)
ALPHA = [
    np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=complex),
    np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]], dtype=complex),
    np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]], dtype=complex),
]
BETA = np.diag([1, 1, -1, -1]).astype(complex)
NU = np.sqrt(7) / 4
I4 = np.eye(4, dtype=complex)

BC_HELIX_R = [1, 3, 0, 2]
BC_HELIX_L = [0, 1, 2, 3]


def make_tau(dirs, a):
    """Construct τ_a from local directions dirs (numpy)."""
    d = dirs[a]
    return NU * BETA + 0.75 * sum(d[i] * ALPHA[i] for i in range(3))


def make_tau_from_dir(d):
    """Construct τ from a single direction vector d (numpy array)."""
    return NU * BETA + 0.75 * sum(d[i] * ALPHA[i] for i in range(3))


def frame_transport(tau_from, tau_to):
    """
    Construct the unitary mapping eigenspaces of tau_from to tau_to.

    Uses the polar decomposition of the overlap matrix
    W = P_to^+ P_from^+ + P_to^- P_from^-, giving the unique
    closest unitary that maps ±1 eigenspaces to ±1 eigenspaces.
    """
    P_from_p = 0.5 * (I4 + tau_from)
    P_from_m = 0.5 * (I4 - tau_from)
    P_to_p = 0.5 * (I4 + tau_to)
    P_to_m = 0.5 * (I4 - tau_to)
    W = P_to_p @ P_from_p + P_to_m @ P_from_m
    WdW = W.conj().T @ W
    evals, evecs = np.linalg.eigh(WdW)
    H_inv = evecs @ np.diag(
        1.0 / np.sqrt(np.maximum(evals, 1e-15))
    ) @ evecs.conj().T
    return W @ H_inv


def vertices_numpy():
    """Return tetrahedral vertex directions as numpy arrays."""
    from src.tetrahedron import vertices as sym_verts
    from sympy import N as symN
    return [np.array([float(symN(v[i])) for i in range(3)]) for v in sym_verts]


def build_helix_taus(N, pattern=None):
    """
    Compute the τ operators at each site along a BC helix of N sites.

    Delegates to helix_geometry.build_taus which uses the analytic vertex
    formula — no sequential reflections, no accumulated numerical error.

    Parameters
    ----------
    N : int
        Number of sites.
    pattern : list of int, optional
        Kept for API compatibility (ignored).

    Returns
    -------
    tau_list : list of ndarray (4×4)
        τ operator at each site.
    """
    from src.helix_geometry import build_taus
    taus_arr = build_taus(N)
    return [taus_arr[n] for n in range(N)]


def build_shift_operator(N, tau_list=None, pattern=None):
    """
    Build the shift operator S_R on a BC helix chain with periodic BC.

    The shift uses position-dependent τ projectors with frame transport
    to ensure unitarity:

        S sends P_n^+ ψ(n) to site n+1 (with U_{n→n+1})
              and P_n^- ψ(n) to site n-1 (with U_{n→n-1})

    Parameters
    ----------
    N : int
        Number of sites (periodic boundary conditions).
    tau_list : list of ndarray, optional
        Pre-computed τ operators. If None, computed from pattern.
    pattern : list of int, optional
        Face-exit pattern. Defaults to BC_HELIX_R.

    Returns
    -------
    S : ndarray (4N × 4N)
        The shift operator matrix.
    """
    if tau_list is None:
        tau_list = build_helix_taus(N, pattern)

    S = np.zeros((4 * N, 4 * N), dtype=complex)

    for n in range(N):
        n1 = (n + 1) % N
        n_1 = (n - 1) % N

        U_fwd = frame_transport(tau_list[n], tau_list[n1])
        U_bwd = frame_transport(tau_list[n], tau_list[n_1])
        P_plus = 0.5 * (I4 + tau_list[n])
        P_minus = 0.5 * (I4 - tau_list[n])

        S[n1 * 4:(n1 + 1) * 4, n * 4:(n + 1) * 4] += U_fwd @ P_plus
        S[n_1 * 4:(n_1 + 1) * 4, n * 4:(n + 1) * 4] += U_bwd @ P_minus

    return S


def analyze_spectrum(S, N):
    """
    Analyze the spectrum of a shift operator.

    Returns
    -------
    dict with keys:
        'eigenvalues': complex array
        'phases': real array (quasi-energies)
        'unitarity_error': float
        'bands': list of arrays (sorted positive quasi-energies, split by band)
    """
    evals = np.linalg.eigvals(S)
    phases = np.angle(evals)

    SdS = S.conj().T @ S
    unitarity_err = np.linalg.norm(SdS - np.eye(4 * N))

    # Extract positive phases, sort, remove degeneracy
    pos_phases = sorted([p for p in phases if p > 0.001])
    # Remove approximate duplicates (degeneracy)
    unique_pos = []
    for p in pos_phases:
        if not unique_pos or abs(p - unique_pos[-1]) > 0.001:
            unique_pos.append(p)

    # Split into bands (alternating)
    band_A = unique_pos[0::2] if len(unique_pos) > 1 else unique_pos
    band_B = unique_pos[1::2] if len(unique_pos) > 1 else []

    return {
        'eigenvalues': evals,
        'phases': phases,
        'unitarity_error': unitarity_err,
        'bands': [np.array(band_A), np.array(band_B)],
    }
