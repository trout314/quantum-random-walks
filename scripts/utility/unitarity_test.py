#!/usr/bin/env python3
"""
Test unitarity of the shift operator on a closed chain.

Creates a small closed R-chain (loop), puts random amplitude on it,
applies the shift, and checks if ||ψ||² is conserved.

This isolates the shift operator from boundary effects (absorption, pruning,
extension) to check if the core walk step is unitary.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from helix_geometry import build_taus, centroid, exit_direction, vertex
from manifold_walk import make_tau, frame_transport

def proj_plus(tau):
    return 0.5 * (np.eye(4, dtype=complex) + tau)

def proj_minus(tau):
    return 0.5 * (np.eye(4, dtype=complex) - tau)


def test_shift_closed_chain(N, verbose=True):
    """Test shift unitarity on a closed chain of length N.

    Uses the BC helix exit directions for the first N sites,
    then manually closes the chain (site N-1 → site 0).
    """
    # Build exit directions for N sites along a BC helix chain
    taus = build_taus(N + 5)  # extra for safety
    dirs = []
    for k in range(N):
        d = exit_direction(k)
        dirs.append(d / np.linalg.norm(d))

    # Build τ and projectors for each site
    tau_ops = [make_tau(d) for d in dirs]
    Pp = [proj_plus(t) for t in tau_ops]
    Pm = [proj_minus(t) for t in tau_ops]

    # Frame transport matrices
    U_fwd = []  # U_fwd[i] transports from site i to site (i+1) % N
    U_bwd = []  # U_bwd[i] transports from site i to site (i-1) % N
    for i in range(N):
        j_fwd = (i + 1) % N
        j_bwd = (i - 1) % N
        U_fwd.append(frame_transport(tau_ops[i], tau_ops[j_fwd]))
        U_bwd.append(frame_transport(tau_ops[i], tau_ops[j_bwd]))

    # Random initial state
    rng = np.random.default_rng(42)
    psi = [rng.standard_normal(4) + 1j * rng.standard_normal(4) for _ in range(N)]
    # Normalize
    norm0 = sum(np.real(np.vdot(p, p)) for p in psi)
    for i in range(N):
        psi[i] /= np.sqrt(norm0)
    norm0 = 1.0

    if verbose:
        print(f"Closed chain with {N} sites")
        print(f"  Initial norm: {sum(np.real(np.vdot(p, p)) for p in psi):.15f}")

    # Apply shift: P+ component goes forward, P- goes backward
    # new_psi[j] = sum of transported contributions arriving at j
    new_psi = [np.zeros(4, dtype=complex) for _ in range(N)]

    for i in range(N):
        j_fwd = (i + 1) % N
        j_bwd = (i - 1) % N

        psi_plus = Pp[i] @ psi[i]   # forward component
        psi_minus = Pm[i] @ psi[i]  # backward component

        # Transport P+ to next site
        new_psi[j_fwd] += frame_transport(tau_ops[i], tau_ops[j_fwd]) @ psi_plus
        # Transport P- to prev site
        new_psi[j_bwd] += frame_transport(tau_ops[i], tau_ops[j_bwd]) @ psi_minus

    norm1 = sum(np.real(np.vdot(p, p)) for p in new_psi)

    if verbose:
        print(f"  After shift:  {norm1:.15f}")
        print(f"  Δnorm:        {norm1 - 1.0:.2e}")

    return norm1


def test_vmix_unitarity(N, phi=0.05, verbose=True):
    """Test V-mixing unitarity at a single site."""
    d = exit_direction(0)
    d = d / np.linalg.norm(d)
    tau = make_tau(d)
    Pp = proj_plus(tau)
    Pm = proj_minus(tau)

    # Gram-Schmidt for P+ and P- bases
    def get_basis(P):
        basis = []
        for col in range(4):
            if len(basis) >= 2:
                break
            v = P[:, col].copy()
            for b in basis:
                v -= np.vdot(b, v) * b
            if np.linalg.norm(v) > 1e-10:
                basis.append(v / np.linalg.norm(v))
        return basis

    pp_basis = get_basis(Pp)
    pm_basis = get_basis(Pm)

    M = np.zeros((4, 4), dtype=complex)
    for j in range(min(len(pp_basis), len(pm_basis))):
        M += np.outer(pm_basis[j], pp_basis[j].conj())
        M += np.outer(pp_basis[j], pm_basis[j].conj())

    V = np.cos(phi) * np.eye(4) + 1j * np.sin(phi) * M

    # Check V is unitary
    VVdag = V @ V.conj().T
    err = np.linalg.norm(VVdag - np.eye(4))
    if verbose:
        print(f"\nV-mixing unitarity (φ={phi}):")
        print(f"  ||V V† - I|| = {err:.2e}")
    return err


def test_full_step_closed(N, phi=0.05, n_steps=100, verbose=True):
    """Test full walk step (shift + vmix) on a closed chain."""
    dirs = []
    for k in range(N):
        d = exit_direction(k)
        dirs.append(d / np.linalg.norm(d))

    tau_ops = [make_tau(d) for d in dirs]
    Pp = [proj_plus(t) for t in tau_ops]
    Pm = [proj_minus(t) for t in tau_ops]

    def get_basis(P):
        basis = []
        for col in range(4):
            if len(basis) >= 2:
                break
            v = P[:, col].copy()
            for b in basis:
                v -= np.vdot(b, v) * b
            if np.linalg.norm(v) > 1e-10:
                basis.append(v / np.linalg.norm(v))
        return basis

    # Precompute V-mixing operators
    V_ops = []
    for i in range(N):
        pp_basis = get_basis(Pp[i])
        pm_basis = get_basis(Pm[i])
        M = np.zeros((4, 4), dtype=complex)
        for j in range(min(len(pp_basis), len(pm_basis))):
            M += np.outer(pm_basis[j], pp_basis[j].conj())
            M += np.outer(pp_basis[j], pm_basis[j].conj())
        V = np.cos(phi) * np.eye(4) + 1j * np.sin(phi) * M
        V_ops.append(V)

    # Random IC
    rng = np.random.default_rng(42)
    psi = [rng.standard_normal(4) + 1j * rng.standard_normal(4) for _ in range(N)]
    norm0 = sum(np.real(np.vdot(p, p)) for p in psi)
    for i in range(N):
        psi[i] /= np.sqrt(norm0)

    if verbose:
        print(f"\nFull walk (shift + V-mix) on closed chain, N={N}, φ={phi}")
        print(f"{'step':>5} {'norm':>18} {'Δnorm':>14}")

    for t in range(n_steps):
        norm_t = sum(np.real(np.vdot(p, p)) for p in psi)
        if verbose and (t < 10 or t % 10 == 0):
            print(f"{t:5d} {norm_t:18.15f} {norm_t - 1.0:14.2e}")

        # Shift
        new_psi = [np.zeros(4, dtype=complex) for _ in range(N)]
        for i in range(N):
            j_fwd = (i + 1) % N
            j_bwd = (i - 1) % N
            psi_plus = Pp[i] @ psi[i]
            psi_minus = Pm[i] @ psi[i]
            new_psi[j_fwd] += frame_transport(tau_ops[i], tau_ops[j_fwd]) @ psi_plus
            new_psi[j_bwd] += frame_transport(tau_ops[i], tau_ops[j_bwd]) @ psi_minus
        psi = new_psi

        # V-mix
        for i in range(N):
            psi[i] = V_ops[i] @ psi[i]

    norm_final = sum(np.real(np.vdot(p, p)) for p in psi)
    if verbose:
        print(f"{n_steps:5d} {norm_final:18.15f} {norm_final - 1.0:14.2e}")

    return norm_final


if __name__ == '__main__':
    print("=" * 60)
    print("UNITARITY TESTS")
    print("=" * 60)

    # Test 1: Shift on closed chains of various lengths
    print("\n--- Test 1: Shift operator on closed chains ---")
    for N in [5, 10, 20, 27, 50]:
        test_shift_closed_chain(N)

    # Test 2: V-mixing unitarity
    test_vmix_unitarity(10, phi=0.05)
    test_vmix_unitarity(10, phi=0.3)

    # Test 3: Full walk step on closed chain
    test_full_step_closed(27, phi=0.05, n_steps=100)
    test_full_step_closed(27, phi=0.0, n_steps=100)
