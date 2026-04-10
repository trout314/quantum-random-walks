#!/usr/bin/env python3
"""
Test E: Momentum eigenstates vs walk eigenstates on a periodic chain.

The 1D BC-helix walk has no translational symmetry (irrational screw angle),
so plane-wave momentum states |k,s⟩ = N^{-1/2} Σ_n e^{ikn} |n⟩⊗|s⟩
should NOT be eigenstates of the walk operator W.

This script:
  1. Builds W = V · S (periodic BC, frame transport glues ends)
  2. Diagonalizes W → true eigenstates {|ψ_j⟩} with phases E_j
  3. Constructs plane-wave momentum states for each allowed k = 2πm/N
  4. Expands each |k,s⟩ in the walk eigenbasis
  5. Quantifies how many walk eigenstates each momentum state overlaps
     (participation ratio in the eigenbasis)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.walk import build_shift_operator, build_helix_taus, I4, frame_transport


def build_vmix(N, taus, phi):
    """Build the V-mixing operator (block diagonal, 4N×4N)."""
    dim = 4 * N
    V = np.eye(dim, dtype=complex)
    if phi == 0:
        return V
    cp, sp = np.cos(phi), np.sin(phi)
    for n in range(N):
        Pp = 0.5 * (I4 + taus[n])
        Pm = 0.5 * (I4 - taus[n])
        # Gram-Schmidt for P+ and P- eigenspace bases
        pp_basis = np.zeros((4, 2), dtype=complex)
        pm_basis = np.zeros((4, 2), dtype=complex)
        np_found = nm_found = 0
        for col in range(4):
            if np_found >= 2 and nm_found >= 2:
                break
            if np_found < 2:
                v = Pp[:, col].copy()
                for j in range(np_found):
                    v -= np.vdot(pp_basis[:, j], v) * pp_basis[:, j]
                nm = np.real(np.vdot(v, v))
                if nm > 1e-10:
                    pp_basis[:, np_found] = v / np.sqrt(nm)
                    np_found += 1
            if nm_found < 2:
                v = Pm[:, col].copy()
                for j in range(nm_found):
                    v -= np.vdot(pm_basis[:, j], v) * pm_basis[:, j]
                nm = np.real(np.vdot(v, v))
                if nm > 1e-10:
                    pm_basis[:, nm_found] = v / np.sqrt(nm)
                    nm_found += 1
        M = np.zeros((4, 4), dtype=complex)
        for j in range(2):
            M += np.outer(pm_basis[:, j], pp_basis[:, j].conj())
            M += np.outer(pp_basis[:, j], pm_basis[:, j].conj())
        Vmix = cp * I4 + 1j * sp * M
        V[4*n:4*n+4, 4*n:4*n+4] = Vmix
    return V


def build_walk_periodic(N, taus, phi):
    """Build walk operator W = V · S on a periodic chain of N sites."""
    S = build_shift_operator(N, taus)
    V = build_vmix(N, taus, phi)
    return V @ S


def build_momentum_states(N):
    """
    Build plane-wave momentum states on N sites with 4 spinor components.

    Returns (k_vals, states) where:
      k_vals: array of N allowed momenta 2πm/N, m = 0,...,N-1
      states: (N, 4, 4N) — states[m, s, :] = |k_m, s⟩  (normalized)
    """
    k_vals = 2 * np.pi * np.arange(N) / N
    states = np.zeros((N, 4, 4 * N), dtype=complex)
    for m in range(N):
        for s in range(4):
            for n in range(N):
                states[m, s, 4*n + s] = np.exp(1j * k_vals[m] * n) / np.sqrt(N)
    return k_vals, states


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08

    print(f"=== Test E: Momentum eigenstates on periodic chain ===")
    print(f"N = {N} sites,  phi = {phi},  Hilbert space dim = {4*N}")

    # --- Build and diagonalize walk operator ---
    print("\nBuilding tau operators...")
    taus = build_helix_taus(N)

    print("Building walk operator W = V·S (periodic BC)...")
    W = build_walk_periodic(N, taus, phi)

    # Verify unitarity
    err = np.linalg.norm(W.conj().T @ W - np.eye(4*N))
    print(f"Unitarity check: ||W†W - I|| = {err:.2e}")

    print("Diagonalizing W...")
    eigenvalues, eigvecs = np.linalg.eig(W)
    E = np.angle(eigenvalues)
    order = np.argsort(E)
    E = E[order]
    eigvecs = eigvecs[:, order]

    # --- Build momentum states ---
    print("Building momentum eigenstates |k,s⟩...")
    k_vals, mom_states = build_momentum_states(N)

    # --- Project momentum states onto walk eigenbasis ---
    # overlap[m, s, j] = |⟨ψ_j | k_m, s⟩|²
    print("Computing overlaps...")
    # For each (m, s), compute the full overlap vector with all eigenstates
    # This is |eigvecs† · |k,s⟩|² for each component

    # eigvecs columns are walk eigenstates; overlap = eigvecs† @ mom_state
    # Do it efficiently: eigvecs is (4N, 4N), mom_states[m,s] is (4N,)
    eigvecs_dag = eigvecs.conj().T  # (4N, 4N)

    # Participation ratio in walk eigenbasis for each |k,s⟩
    # PR = 1 / Σ_j |⟨ψ_j|k,s⟩|⁴  (= 1 if eigenstate, = dim if uniform)
    pr_matrix = np.zeros((N, 4))  # PR for each (k, spinor)
    max_overlap = np.zeros((N, 4))  # largest single |⟨ψ_j|k,s⟩|²

    for m in range(N):
        for s in range(4):
            coeffs = eigvecs_dag @ mom_states[m, s]
            probs = np.abs(coeffs)**2
            pr_matrix[m, s] = 1.0 / np.sum(probs**2)
            max_overlap[m, s] = np.max(probs)

    # Average participation ratio across spinor components
    pr_avg = np.mean(pr_matrix, axis=1)
    max_ov_avg = np.mean(max_overlap, axis=1)

    # --- Wrap k to [-π, π] for nicer plotting ---
    k_plot = np.where(k_vals > np.pi, k_vals - 2*np.pi, k_vals)
    k_order = np.argsort(k_plot)
    k_plot = k_plot[k_order]
    pr_avg = pr_avg[k_order]
    max_ov_avg = max_ov_avg[k_order]
    pr_matrix_sorted = pr_matrix[k_order]

    # --- Statistics ---
    print(f"\n--- Participation ratio of |k,s⟩ in walk eigenbasis ---")
    print(f"  (PR = 1 means |k,s⟩ IS a walk eigenstate)")
    print(f"  (PR = 4N = {4*N} means |k,s⟩ overlaps all eigenstates equally)")
    print(f"  Min PR (across k, averaged over s): {pr_avg.min():.1f}")
    print(f"  Max PR: {pr_avg.max():.1f}")
    print(f"  Mean PR: {pr_avg.mean():.1f}")
    print(f"  Max single overlap |⟨ψ_j|k,s⟩|² (best case): {max_ov_avg.max():.4f}")
    print(f"  Mean max overlap: {max_ov_avg.mean():.4f}")

    if pr_avg.min() > 1.5:
        print(f"\n  ✓ CONFIRMED: No momentum eigenstate is a walk eigenstate")
        print(f"    (all PR > 1.5, i.e., every |k⟩ spreads over multiple walk eigenstates)")
    else:
        print(f"\n  ⚠ Some momentum eigenstates are close to walk eigenstates (PR ≈ 1)")

    # --- Detailed view: pick a few k values, show their eigenbasis decomposition ---
    # Choose k ≈ 0, π/4, π/2, 3π/4, π
    target_ks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

    # --- PLOTTING ---
    fig = plt.figure(figsize=(18, 14))

    # Panel 1: PR vs k
    ax1 = fig.add_subplot(2, 3, 1)
    for s in range(4):
        ax1.plot(k_plot, pr_matrix_sorted[:, s], '.', markersize=3, alpha=0.5,
                 label=f'spinor {s}')
    ax1.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='PR=1 (eigenstate)')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Participation ratio')
    ax1.set_title('PR of |k,s⟩ in walk eigenbasis')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Max overlap vs k
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(k_plot, max_ov_avg, '.', markersize=3, color='darkblue')
    ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='max=1 (eigenstate)')
    ax2.set_xlabel('k')
    ax2.set_ylabel(r'max$_j$ $|\langle\psi_j|k\rangle|^2$')
    ax2.set_title('Best overlap with any single walk eigenstate')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Walk eigenspectrum
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=3, alpha=0.5)
    theta_c = np.linspace(0, 2*np.pi, 200)
    ax3.plot(np.cos(theta_c), np.sin(theta_c), 'k-', alpha=0.2)
    ax3.set_aspect('equal')
    ax3.set_xlabel('Re(λ)')
    ax3.set_ylabel('Im(λ)')
    ax3.set_title('Walk eigenvalues on unit circle')
    ax3.grid(True, alpha=0.3)

    # Panels 4-6: Eigenbasis decomposition for selected k values
    # For each selected k, show |⟨ψ_j|k,s=0⟩|² vs E_j
    k_vals_orig = 2 * np.pi * np.arange(N) / N
    for idx, k_target in enumerate(target_ks[:3]):
        ax = fig.add_subplot(2, 3, 4 + idx)
        # Find closest k
        m_best = np.argmin(np.abs(k_vals_orig - k_target))
        k_actual = k_vals_orig[m_best]

        for s in range(4):
            coeffs = eigvecs_dag @ mom_states[m_best, s]
            probs = np.abs(coeffs)**2
            ax.plot(E, probs, '-', alpha=0.6, linewidth=0.8, label=f's={s}')

        ax.set_xlabel('Walk quasienergy E')
        ax.set_ylabel(r'$|\langle\psi_j|k,s\rangle|^2$')
        k_disp = k_actual if k_actual <= np.pi else k_actual - 2*np.pi
        ax.set_title(f'|k={k_disp:.2f}, s⟩ decomposition')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Test E: Momentum states are NOT walk eigenstates\n'
        f'N={N}, φ={phi}, periodic BC with frame transport',
        fontsize=14
    )
    plt.tight_layout()

    out = '/tmp/test_E_momentum_eigenstates.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    # --- Also: time evolution of a momentum eigenstate ---
    # Show that |k⟩ changes over time (not stationary)
    print("\n--- Time evolution of a momentum eigenstate ---")
    m_test = N // 8  # pick k ≈ π/4
    k_test = k_vals_orig[m_test]
    psi0 = mom_states[m_test, 0].copy()  # |k, s=0⟩

    steps = [0, 1, 5, 20, 50]
    fig2, axes2 = plt.subplots(1, len(steps), figsize=(4*len(steps), 4))

    psi = psi0.copy()
    step_idx = 0
    for t in range(max(steps) + 1):
        if t in steps:
            density = np.sum(np.abs(psi.reshape(N, 4))**2, axis=1)
            ax = axes2[step_idx]
            ax.bar(range(N), density, width=1.0, alpha=0.7)
            # For reference, show the flat distribution of a momentum eigenstate
            ax.axhline(1.0/N, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Site n')
            ax.set_ylabel('P(n)')
            ax.set_title(f't = {t}')
            ax.set_ylim(0, max(5.0/N, density.max()*1.2))
            step_idx += 1
        if t < max(steps):
            psi = W @ psi

    k_disp = k_test if k_test <= np.pi else k_test - 2*np.pi
    fig2.suptitle(
        f'Time evolution of |k={k_disp:.2f}, s=0⟩ under W\n'
        f'N={N}, φ={phi} — a true eigenstate would stay flat',
        fontsize=12
    )
    plt.tight_layout()

    out2 = '/tmp/test_E_momentum_evolution.png'
    plt.savefig(out2, dpi=150)
    print(f"Time evolution plot saved to {out2}")


if __name__ == '__main__':
    main()
