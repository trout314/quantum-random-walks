#!/usr/bin/env python3
"""
Construct walk-optimal ICs using the quasi-Bloch framework.

The key insight: walk eigenstates at momentum k are NOT pure plane waves
e^{ikn} χ — they're dressed plane waves:

  ψ_k(n) = e^{ikn} Σ_m c_{m,s}(k) e^{imnθ}

where θ = arccos(-2/3) and c(k) is the eigenvector of the small W(k) matrix.

The m=0 component c_{0,s}(k) is the "walk-optimal spinor" at momentum k.
The m≠0 components give the quasiperiodic dressing.

This script:
1. Computes W(k) eigenstates to get the walk spinor χ_walk(k) vs Dirac spinor χ_Dirac(k)
2. Builds a Gaussian wavepacket using the correct walk spinors
3. Compares with Dirac-frame ICs to quantify the mismatch
4. Exports the quasi-Bloch IC for use in simulations
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helix_geometry import build_taus

BETA = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)

def proj_plus(tau):
    return 0.5 * (np.eye(4) + tau)

def proj_minus(tau):
    return 0.5 * (np.eye(4) - tau)

def frame_transport(tau_from, tau_to):
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    cos_half = np.sqrt(max((1 + cos_theta) / 2, 1e-15))
    scale = 1 / (2 * cos_half)
    return scale * (np.eye(4) + prod)

THETA_BC = np.arccos(-2/3)


def build_vmix_block(tau, mix_phi):
    if mix_phi == 0:
        return np.eye(4, dtype=complex)
    cp, sp = np.cos(mix_phi), np.sin(mix_phi)
    Pp = proj_plus(tau)
    Pm = proj_minus(tau)
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
    return cp * np.eye(4) + 1j * sp * M


# ---- Walk blocks and Fourier coefficients ----

def compute_walk_blocks(N, taus, mix_phi):
    F = np.zeros((N, 4, 4), dtype=complex)
    G = np.zeros((N, 4, 4), dtype=complex)
    for m in range(N):
        V_m = build_vmix_block(taus[m], mix_phi)
        m_prev = (m - 1) % N
        m_next = (m + 1) % N
        U_fwd = frame_transport(taus[m_prev], taus[m])
        U_bwd = frame_transport(taus[m_next], taus[m])
        Pp_prev = proj_plus(taus[m_prev])
        Pm_next = proj_minus(taus[m_next])
        F[m] = V_m @ U_fwd @ Pp_prev
        G[m] = V_m @ U_bwd @ Pm_next
    return F, G


def fourier_coefficients_1d(blocks, N, n_harmonics):
    """1D Fourier coefficients (quasiperiodic only, no sublattice)."""
    M = n_harmonics
    B_hat = np.zeros((2 * M + 1, 4, 4), dtype=complex)
    ns = np.arange(N)
    for l_idx, l in enumerate(range(-M, M + 1)):
        exp_factor = np.exp(-1j * l * ns * THETA_BC)
        for i in range(4):
            for j in range(4):
                B_hat[l_idx, i, j] = np.mean(blocks[:, i, j] * exp_factor)
    return B_hat


def build_Wk_simple(k, F_hat, G_hat, M):
    """Build W(k) in the simple (m, s) basis — no sublattice splitting."""
    dim = 4 * (2 * M + 1)
    W = np.zeros((dim, dim), dtype=complex)
    for m_prime in range(-M, M + 1):
        for m in range(-M, M + 1):
            delta_m = m_prime - m
            l_idx = delta_m + M
            if 0 <= l_idx < 2 * M + 1:
                row = (m_prime + M) * 4
                col = (m + M) * 4
                W[row:row+4, col:col+4] += (
                    np.exp(-1j * k) * F_hat[l_idx] +
                    np.exp(1j * k) * G_hat[l_idx]
                )
    return W


# ---- Full walk operator for testing ----

def build_full_walk(N, taus, mix_phi):
    dim = 4 * N
    S = np.zeros((dim, dim), dtype=complex)
    for n in range(N):
        n_fwd = (n + 1) % N
        n_bwd = (n - 1) % N
        Pp = proj_plus(taus[n])
        Pm = proj_minus(taus[n])
        U_fwd = frame_transport(taus[n], taus[n_fwd])
        U_bwd = frame_transport(taus[n], taus[n_bwd])
        S[4*n_fwd:4*n_fwd+4, 4*n:4*n+4] += U_fwd @ Pp
        S[4*n_bwd:4*n_bwd+4, 4*n:4*n+4] += U_bwd @ Pm
    V = np.eye(dim, dtype=complex)
    if mix_phi != 0:
        for n in range(N):
            V[4*n:4*n+4, 4*n:4*n+4] = build_vmix_block(taus[n], mix_phi)
    return V @ S


# ---- Dirac spinors for comparison ----

def dirac_spinor_positive_energy(k, m_mass):
    """
    Positive-energy Dirac spinor at momentum k (1D, along α₃).

    H = α₃ k + β m → E = √(k² + m²)

    Eigenstate: u(k) ∝ (E + m, 0, k, 0)^T (up spin)
    (In Dirac representation with α₃ = [[0,σ₃],[σ₃,0]], β = [[I,0],[0,-I]])
    """
    E = np.sqrt(k**2 + m_mass**2)
    # For α₃: [0,0,1,0; 0,0,0,-1; 1,0,0,0; 0,-1,0,0]
    # Positive energy, spin-up eigenstate
    u = np.array([E + m_mass, 0, k, 0], dtype=complex)
    u /= np.linalg.norm(u)
    return u


# ---- Main ----

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    m_dirac = 0.9 * mix_phi  # effective Dirac mass

    print(f"N={N}, φ_mix={mix_phi}, M={M}")
    print(f"Effective Dirac mass: m = {m_dirac:.4f}")

    print("Building chain...")
    taus = build_taus(N)

    print("Computing walk blocks and Fourier coefficients...")
    F_blocks, G_blocks = compute_walk_blocks(N, taus, mix_phi)
    F_hat = fourier_coefficients_1d(F_blocks, N, M)
    G_hat = fourier_coefficients_1d(G_blocks, N, M)

    # ============================================================
    # Part 1: Compare walk spinor vs Dirac spinor at each k
    # ============================================================
    print("\n=== Walk vs Dirac spinor comparison ===")

    n_k = 100
    k_points = np.linspace(0, np.pi, n_k)

    walk_spinors = np.zeros((n_k, 4), dtype=complex)  # m=0 component
    walk_energies = np.zeros(n_k)
    dirac_spinors = np.zeros((n_k, 4), dtype=complex)
    overlaps = np.zeros(n_k)  # |<walk|dirac>|²

    for i, k in enumerate(k_points):
        Wk = build_Wk_simple(k, F_hat, G_hat, M)
        evals, evecs = np.linalg.eig(Wk)
        E = np.angle(evals)

        # Find the positive-energy state closest to the Dirac prediction
        E_dirac = np.sqrt(k**2 + m_dirac**2)
        pos_mask = E > 0.001
        if pos_mask.any():
            # Pick the eigenvalue closest to the expected Dirac energy
            diffs = np.abs(E[pos_mask] - E_dirac)
            best_idx = np.where(pos_mask)[0][np.argmin(diffs)]
        else:
            best_idx = np.argmin(np.abs(E - E_dirac))

        walk_energies[i] = E[best_idx]

        # Extract the m=0 spinor component from the eigenvector
        evec = evecs[:, best_idx]
        # The eigenvector has shape 4*(2M+1). Index (m, s) → (m+M)*4 + s
        # m=0 component:
        m0_start = M * 4  # m=0 starts at index M*4
        walk_spinor = evec[m0_start:m0_start+4].copy()
        walk_spinor /= np.linalg.norm(walk_spinor) if np.linalg.norm(walk_spinor) > 1e-10 else 1.0
        walk_spinors[i] = walk_spinor

        # Dirac spinor at this k
        dirac_sp = dirac_spinor_positive_energy(k, m_dirac)
        dirac_spinors[i] = dirac_sp

        # Overlap (gauge-invariant: maximize over global phase)
        overlap = np.abs(np.vdot(walk_spinor, dirac_sp))**2
        overlaps[i] = overlap

    print(f"Overlap |<walk|Dirac>|² at k=0: {overlaps[0]:.6f}")
    print(f"Overlap at k=π/4: {overlaps[n_k//4]:.6f}")
    print(f"Overlap at k=π/2: {overlaps[n_k//2]:.6f}")
    print(f"Min overlap: {overlaps.min():.6f} at k={k_points[np.argmin(overlaps)]:.3f}")

    # ============================================================
    # Part 2: How much weight is in m≠0 harmonics?
    # ============================================================
    print("\n=== Harmonic content of walk eigenstates ===")

    m0_weights = np.zeros(n_k)
    m_nonzero_weights = np.zeros(n_k)

    for i, k in enumerate(k_points):
        Wk = build_Wk_simple(k, F_hat, G_hat, M)
        evals, evecs = np.linalg.eig(Wk)
        E = np.angle(evals)
        E_dirac = np.sqrt(k**2 + m_dirac**2)
        pos_mask = E > 0.001
        if pos_mask.any():
            diffs = np.abs(E[pos_mask] - E_dirac)
            best_idx = np.where(pos_mask)[0][np.argmin(diffs)]
        else:
            best_idx = np.argmin(np.abs(E - E_dirac))

        evec = evecs[:, best_idx]

        # Weight in each harmonic
        total_weight = np.sum(np.abs(evec)**2)
        for m_h in range(-M, M + 1):
            start = (m_h + M) * 4
            weight = np.sum(np.abs(evec[start:start+4])**2) / total_weight
            if m_h == 0:
                m0_weights[i] = weight
            else:
                m_nonzero_weights[i] += weight

    print(f"m=0 weight at k=0: {m0_weights[0]:.6f}")
    print(f"m≠0 weight at k=0: {m_nonzero_weights[0]:.6f}")
    print(f"m=0 weight at k=π/2: {m0_weights[n_k//2]:.6f}")
    print(f"m≠0 weight at k=π/2: {m_nonzero_weights[n_k//2]:.6f}")

    # ============================================================
    # Part 3: Build quasi-Bloch IC and test against full walk
    # ============================================================
    print("\n=== Wavepacket construction test ===")
    sigma = 50
    center = N // 2
    xs = np.arange(N) - center

    # Method A: Dirac-frame IC (baseline)
    gauss = np.exp(-xs**2 / (2 * sigma**2))
    gauss /= np.linalg.norm(gauss)
    chi_dirac_k0 = dirac_spinor_positive_energy(0, m_dirac)
    psi_dirac = np.zeros(4 * N, dtype=complex)
    for n in range(N):
        psi_dirac[4*n:4*n+4] = gauss[n] * chi_dirac_k0
    psi_dirac /= np.linalg.norm(psi_dirac)

    # Method B: Walk-optimal IC via quasi-Bloch
    # Use k=0 walk eigenvector, replicate with quasiperiodic dressing
    Wk0 = build_Wk_simple(0, F_hat, G_hat, M)
    evals0, evecs0 = np.linalg.eig(Wk0)
    E0 = np.angle(evals0)
    E_dirac0 = m_dirac  # E(k=0) = m
    pos_mask = E0 > 0.001
    if pos_mask.any():
        diffs = np.abs(E0[pos_mask] - E_dirac0)
        best_idx = np.where(pos_mask)[0][np.argmin(diffs)]
    else:
        best_idx = np.argmin(np.abs(E0 - E_dirac0))

    evec0 = evecs0[:, best_idx]

    print(f"Walk eigenstate at k=0:")
    print(f"  Energy: E = {E0[best_idx]:.6f} (Dirac: {E_dirac0:.6f})")
    print(f"  Eigenvector harmonic weights:")
    for m_h in range(-M, M + 1):
        start = (m_h + M) * 4
        w = np.sum(np.abs(evec0[start:start+4])**2)
        spinor = evec0[start:start+4]
        print(f"    m={m_h:+d}: weight={w:.6f}  spinor=({spinor[0]:.4f}, {spinor[1]:.4f}, {spinor[2]:.4f}, {spinor[3]:.4f})")

    # Construct quasi-Bloch wavepacket
    psi_bloch = np.zeros(4 * N, dtype=complex)
    for n in range(N):
        phi_n = n * THETA_BC
        # Sum over harmonics
        spinor = np.zeros(4, dtype=complex)
        for m_h in range(-M, M + 1):
            start = (m_h + M) * 4
            spinor += evec0[start:start+4] * np.exp(1j * m_h * phi_n)
        psi_bloch[4*n:4*n+4] = gauss[n] * spinor
    psi_bloch /= np.linalg.norm(psi_bloch)

    # Method C: Walk-optimal m=0 only (constant spinor, no dressing)
    chi_walk_k0 = walk_spinors[0]  # m=0 spinor at k=0
    psi_walk_m0 = np.zeros(4 * N, dtype=complex)
    for n in range(N):
        psi_walk_m0[4*n:4*n+4] = gauss[n] * chi_walk_k0
    psi_walk_m0 /= np.linalg.norm(psi_walk_m0)

    # Evolve all three and compare drift/spreading
    print("\nEvolving wavepackets...")
    W_full = build_full_walk(N, taus, mix_phi)

    n_steps = 200
    metrics = {'dirac': [], 'bloch': [], 'walk_m0': []}

    psi_d = psi_dirac.copy()
    psi_b = psi_bloch.copy()
    psi_m = psi_walk_m0.copy()

    for t in range(n_steps + 1):
        for label, psi in [('dirac', psi_d), ('bloch', psi_b), ('walk_m0', psi_m)]:
            prob = np.array([np.sum(np.abs(psi[4*n:4*n+4])**2) for n in range(N)])
            prob /= prob.sum()
            mean_x = np.sum(prob * xs)
            var_x = np.sum(prob * (xs - mean_x)**2)
            metrics[label].append((mean_x, np.sqrt(var_x)))

        if t < n_steps:
            psi_d = W_full @ psi_d
            psi_b = W_full @ psi_b
            psi_m = W_full @ psi_m

    print(f"\nAfter {n_steps} steps:")
    for label in ['dirac', 'bloch', 'walk_m0']:
        x0, w0 = metrics[label][0]
        xf, wf = metrics[label][-1]
        print(f"  {label:10s}: drift={xf-x0:+.2f}  width: {w0:.1f} → {wf:.1f}")

    # ============================================================
    # Plots
    # ============================================================
    fig = plt.figure(figsize=(18, 14))

    # Panel 1: Walk vs Dirac spinor overlap
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(k_points, overlaps, 'b-', lw=2)
    ax1.set_xlabel('k')
    ax1.set_ylabel('|⟨walk|Dirac⟩|²')
    ax1.set_title('Spinor overlap (walk vs Dirac)')
    ax1.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Harmonic content
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(k_points, m0_weights, 'b-', lw=2, label='m=0 weight')
    ax2.plot(k_points, m_nonzero_weights, 'r-', lw=2, label='m≠0 weight')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Weight')
    ax2.set_title('Harmonic content of walk eigenstates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Walk energy vs Dirac energy
    ax3 = fig.add_subplot(3, 3, 3)
    E_dirac_curve = np.sqrt(k_points**2 + m_dirac**2)
    ax3.plot(k_points, walk_energies, 'b.', ms=3, label='Walk E(k)')
    ax3.plot(k_points, E_dirac_curve, 'r-', lw=2, label='Dirac E(k)')
    ax3.set_xlabel('k')
    ax3.set_ylabel('E')
    ax3.set_title('Dispersion comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Spinor components at k=0
    ax4 = fig.add_subplot(3, 3, 4)
    bar_x = np.arange(4)
    ax4.bar(bar_x - 0.15, np.abs(walk_spinors[0])**2, width=0.3, label='Walk', alpha=0.7)
    ax4.bar(bar_x + 0.15, np.abs(dirac_spinors[0])**2, width=0.3, label='Dirac', alpha=0.7)
    ax4.set_xlabel('Spinor component')
    ax4.set_ylabel('|c_s|²')
    ax4.set_title(f'Spinor at k=0 (overlap={overlaps[0]:.4f})')
    ax4.set_xticks(bar_x)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Spinor components vs k
    ax5 = fig.add_subplot(3, 3, 5)
    for s in range(4):
        ax5.plot(k_points, np.abs(walk_spinors[:, s])**2, '-', lw=1.5,
                label=f'Walk s={s}')
        ax5.plot(k_points, np.abs(dirac_spinors[:, s])**2, '--', lw=1,
                label=f'Dirac s={s}')
    ax5.set_xlabel('k')
    ax5.set_ylabel('|c_s|²')
    ax5.set_title('Spinor components vs k')
    ax5.legend(fontsize=6, ncol=2)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Initial wavepacket densities
    ax6 = fig.add_subplot(3, 3, 6)
    for label, psi_init in [('Dirac', psi_dirac), ('Bloch', psi_bloch), ('Walk m=0', psi_walk_m0)]:
        prob = np.array([np.sum(np.abs(psi_init[4*n:4*n+4])**2) for n in range(N)])
        ax6.plot(xs, prob, label=label, alpha=0.7)
    ax6.set_xlabel('Site (relative to center)')
    ax6.set_ylabel('Probability density')
    ax6.set_title('Initial wavepackets')
    ax6.set_xlim(-3*sigma, 3*sigma)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Panel 7: Drift vs time
    ax7 = fig.add_subplot(3, 3, 7)
    ts = np.arange(n_steps + 1)
    for label, color in [('dirac', 'C0'), ('bloch', 'C1'), ('walk_m0', 'C2')]:
        drift = [m[0] - metrics[label][0][0] for m in metrics[label]]
        ax7.plot(ts, drift, color=color, label=label, lw=1.5)
    ax7.set_xlabel('Time step')
    ax7.set_ylabel('Drift (sites)')
    ax7.set_title(f'Wavepacket drift ({n_steps} steps)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Panel 8: Width vs time
    ax8 = fig.add_subplot(3, 3, 8)
    for label, color in [('dirac', 'C0'), ('bloch', 'C1'), ('walk_m0', 'C2')]:
        widths = [m[1] for m in metrics[label]]
        ax8.plot(ts, widths, color=color, label=label, lw=1.5)
    ax8.set_xlabel('Time step')
    ax8.set_ylabel('Width (σ)')
    ax8.set_title('Wavepacket width')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Panel 9: Final density profiles
    ax9 = fig.add_subplot(3, 3, 9)
    for label, psi_final, color in [('Dirac', psi_d, 'C0'), ('Bloch', psi_b, 'C1'),
                                      ('Walk m=0', psi_m, 'C2')]:
        prob = np.array([np.sum(np.abs(psi_final[4*n:4*n+4])**2) for n in range(N)])
        ax9.plot(xs, prob, color=color, label=label, alpha=0.7)
    ax9.set_xlabel('Site')
    ax9.set_ylabel('Probability density')
    ax9.set_title(f'After {n_steps} steps')
    ax9.set_xlim(-4*sigma, 4*sigma)
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    fig.suptitle(f'Quasi-Bloch IC analysis: N={N}, φ_mix={mix_phi}, σ={sigma}\n'
                 f'Walk spinor from W(k) with M={M} harmonics', fontsize=14)
    plt.tight_layout()

    out = '/tmp/quasi_bloch_ic.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    main()
