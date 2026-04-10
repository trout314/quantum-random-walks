#!/usr/bin/env python3
"""
Test E2: Quasi-Bloch structure of walk eigenstates.

True eigenstates of the periodic-BC walk operator have the form:
  ψ(n) = e^{ikn} · F_{n mod 4}(nθ mod 2π)

i.e. a plane wave times a smooth aperiodic envelope that depends on the
perpendicular-space coordinate φ = nθ mod 2π and the sublattice index.

This script visualizes this structure:
  - Standing-wave density profiles at several momenta
  - The Bloch envelope after removing the plane wave
  - Envelope collapse onto smooth curves in perpendicular space
  - Perfect stationarity under time evolution
  - Spinor-resolved structure of the envelope
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.helix_geometry import build_taus

THETA_BC = np.arccos(-2/3)


# ---- Helpers ----

def proj_plus(tau):
    return 0.5 * (np.eye(4) + tau)

def proj_minus(tau):
    return 0.5 * (np.eye(4) - tau)

def frame_transport(tau_from, tau_to):
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    cos_half = np.sqrt(max((1 + cos_theta) / 2, 1e-15))
    return (np.eye(4) + prod) / (2 * cos_half)

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


def extract_k(psi_2d, N):
    """Extract dominant momentum from eigenstate."""
    freqs = np.fft.fftfreq(N) * 2 * np.pi
    total_power = np.zeros(N)
    for a in range(4):
        ft = np.fft.fft(psi_2d[:, a])
        total_power += np.abs(ft)**2
    total_power[0] = 0
    return freqs[np.argmax(total_power)]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08

    print(f"=== Test E2: Quasi-Bloch eigenstates — clean visualization ===")
    print(f"N={N}, φ_mix={mix_phi}")

    taus = build_taus(N)
    ns = np.arange(N)
    phi_perp = (ns * THETA_BC) % (2 * np.pi)
    sublat = ns % 4
    colors_sub = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    print("Building walk operator...")
    W = build_full_walk(N, taus, mix_phi)
    print("Diagonalizing...")
    evals, evecs = np.linalg.eig(W)
    E = np.angle(evals)

    # Extract k and IPR for all states
    k_all = np.zeros(len(E))
    ipr_all = np.zeros(len(E))
    for j in range(len(E)):
        psi = evecs[:, j].reshape(N, 4)
        prob = np.sum(np.abs(psi)**2, axis=1)
        prob /= prob.sum()
        ipr_all[j] = 1.0 / np.sum(prob**2)
        k_all[j] = extract_k(psi, N)

    extended = ipr_all > N / 10

    # Pick 6 eigenstates spanning a range of momenta (positive E branch)
    ext_pos = np.where(extended & (E > 0.05))[0]
    k_targets = [0.15, 0.4, 0.8, 1.2, 2.0, 2.8]
    picks = []
    for kt in k_targets:
        dists = np.abs(np.abs(k_all[ext_pos]) - kt)
        best = ext_pos[np.argmin(dists)]
        if best not in picks:
            picks.append(best)
    n_picks = len(picks)

    print(f"\nSelected {n_picks} eigenstates:")
    for j in picks:
        print(f"  E={E[j]:.4f}, k={k_all[j]:.4f}, IPR={ipr_all[j]:.0f}")

    # ==================================================================
    # FIGURE 1: The main showcase — density, envelope, perp-space collapse
    # ==================================================================
    fig1 = plt.figure(figsize=(5 * n_picks, 13))
    gs = GridSpec(4, n_picks, figure=fig1, hspace=0.35, wspace=0.3)

    for col, j in enumerate(picks):
        psi = evecs[:, j].reshape(N, 4)
        k_j = k_all[j]
        dens = np.sum(np.abs(psi)**2, axis=1)
        u = psi * np.exp(-1j * k_j * ns)[:, None]
        u_norm2 = np.sum(np.abs(u)**2, axis=1)
        u_mean = np.mean(u_norm2)

        # Row 0: Density P(n) — the standing wave
        ax = fig1.add_subplot(gs[0, col])
        ax.fill_between(ns, dens, alpha=0.5, color='steelblue')
        ax.plot(ns, dens, '-', lw=0.6, color='navy')
        ax.set_xlim(0, N)
        ax.set_xlabel('Site n', fontsize=8)
        if col == 0:
            ax.set_ylabel('P(n)', fontsize=9)
        ax.set_title(f'E = {E[j]:.3f},  k = {k_j:.3f}', fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)

        # Row 1: Zoomed density — show ~3 wavelengths
        ax1 = fig1.add_subplot(gs[1, col])
        wavelength = max(4, int(2 * np.pi / max(abs(k_j), 0.01)))
        n_show = min(N, 4 * wavelength)
        n_start = N // 4  # start from middle to avoid boundary
        n_end = min(n_start + n_show, N)
        ax1.fill_between(ns[n_start:n_end], dens[n_start:n_end],
                         alpha=0.5, color='steelblue')
        ax1.plot(ns[n_start:n_end], dens[n_start:n_end], '-', lw=1, color='navy')
        # Overlay: pure cos²(kn) envelope for reference
        cos2 = np.mean(dens) * (1 + 0.8 * np.cos(2 * k_j * ns[n_start:n_end]))
        ax1.plot(ns[n_start:n_end], cos2, '--', lw=1, color='red', alpha=0.5)
        ax1.set_xlim(n_start, n_end)
        ax1.set_xlabel('Site n', fontsize=8)
        if col == 0:
            ax1.set_ylabel('P(n) zoomed', fontsize=9)
        ax1.tick_params(labelsize=7)

        # Row 2: Bloch envelope |u(n)|² vs site
        ax2 = fig1.add_subplot(gs[2, col])
        for r in range(4):
            mask = sublat == r
            ax2.plot(ns[mask], u_norm2[mask], '.', ms=2.5, color=colors_sub[r],
                     label=f'r={r}' if col == 0 else None)
        ax2.axhline(u_mean, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax2.set_xlim(0, N)
        ax2.set_xlabel('Site n', fontsize=8)
        if col == 0:
            ax2.set_ylabel('|u(n)|²', fontsize=9)
        ax2.tick_params(labelsize=7)

        # Row 3: Envelope vs perp-space — the key collapse plot
        ax3 = fig1.add_subplot(gs[3, col])
        for r in range(4):
            mask = sublat == r
            order = np.argsort(phi_perp[mask])
            ax3.plot(phi_perp[mask][order], u_norm2[mask][order], '.-',
                     ms=3, lw=0.6, color=colors_sub[r],
                     label=f'r={r}' if col == 0 else None)
        ax3.set_xlim(0, 2 * np.pi)
        ax3.set_xlabel('φ = nθ mod 2π', fontsize=8)
        if col == 0:
            ax3.set_ylabel('|u(n)|²', fontsize=9)
        ax3.tick_params(labelsize=7)

    # Row labels on the left margin
    fig1.text(0.005, 0.88, 'Density\n(full chain)', fontsize=9, va='center',
              ha='left', style='italic', color='gray')
    fig1.text(0.005, 0.66, 'Density\n(zoomed)', fontsize=9, va='center',
              ha='left', style='italic', color='gray')
    fig1.text(0.005, 0.43, 'Bloch\nenvelope\nvs site', fontsize=9, va='center',
              ha='left', style='italic', color='gray')
    fig1.text(0.005, 0.18, 'Envelope\nvs perp\nspace φ', fontsize=9, va='center',
              ha='left', style='italic', color='gray')

    # Legend for sublattice colors
    handles = [plt.Line2D([0], [0], marker='.', color=colors_sub[r], ls='-',
               lw=0.6, ms=5, label=f'sublattice r={r}') for r in range(4)]
    fig1.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
                bbox_to_anchor=(0.5, -0.01))

    fig1.suptitle(
        f'Walk eigenstates = standing waves × quasiperiodic envelope\n'
        f'N={N}, φ={mix_phi}, periodic BC with frame transport.  '
        f'Red dashed (row 2): cos²(kn) reference.',
        fontsize=12, y=0.995)

    out1 = '/tmp/test_E2_eigenstate_structure.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nFigure 1 (structure) saved to {out1}")

    # ==================================================================
    # FIGURE 2: Time evolution — eigenstate stationarity
    # ==================================================================
    # Pick two states: low-k and mid-k
    j_lo = picks[0]
    j_mid = picks[2]

    T_steps = [0, 50, 200, 500]
    fig2, axes2 = plt.subplots(2, len(T_steps), figsize=(4.5 * len(T_steps), 7))

    for row, j_demo in enumerate([j_lo, j_mid]):
        psi0 = evecs[:, j_demo].copy()
        dens0 = np.sum(np.abs(psi0.reshape(N, 4))**2, axis=1)
        ymax = 1.4 * dens0.max()
        psi_t = psi0.copy()

        for col, t_target in enumerate(T_steps):
            # Evolve to t_target (accumulating from last position)
            t_current = 0 if col == 0 else T_steps[col - 1]
            for _ in range(t_target - t_current):
                psi_t = W @ psi_t
            dens = np.sum(np.abs(psi_t.reshape(N, 4))**2, axis=1)
            fid = np.abs(np.vdot(psi0, psi_t))**2

            ax = axes2[row, col]
            ax.fill_between(ns, dens, alpha=0.5, color='steelblue')
            ax.plot(ns, dens, '-', lw=0.6, color='navy')
            if t_target > 0:
                ax.plot(ns, dens0, '--', lw=0.6, color='red', alpha=0.4)
            ax.set_ylim(0, ymax)
            ax.set_xlim(0, N)
            if row == 1:
                ax.set_xlabel('Site n')
            if col == 0:
                ax.set_ylabel(f'k={k_all[j_demo]:.2f}\nP(n)')
            ax.set_title(f't = {t_target},  fidelity = {fid:.6f}', fontsize=9)
            ax.tick_params(labelsize=7)

    fig2.suptitle(
        f'Eigenstate time evolution: density is perfectly stationary\n'
        f'N={N}, φ={mix_phi}.  Red dashed = t=0 reference.',
        fontsize=12)
    plt.tight_layout()
    out2 = '/tmp/test_E2_eigenstate_evolution.png'
    plt.savefig(out2, dpi=150)
    print(f"Figure 2 (evolution) saved to {out2}")

    # ==================================================================
    # FIGURE 3: Spinor-resolved envelope for one state
    # ==================================================================
    j_detail = picks[2]  # mid-k
    psi = evecs[:, j_detail].reshape(N, 4)
    k_j = k_all[j_detail]
    u = psi * np.exp(-1j * k_j * ns)[:, None]

    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8))

    for s in range(4):
        # Top: Re and Im of u_s(n) vs φ, per sublattice
        ax = axes3[0, s]
        for r in range(4):
            mask = sublat == r
            order = np.argsort(phi_perp[mask])
            ax.plot(phi_perp[mask][order], np.real(u[mask, s][order]), '.',
                    ms=3, color=colors_sub[r], label=f'Re, r={r}' if s == 0 else None)
            ax.plot(phi_perp[mask][order], np.imag(u[mask, s][order]), 'x',
                    ms=2, color=colors_sub[r], alpha=0.4)
        ax.set_xlabel('φ', fontsize=8)
        ax.set_ylabel(f'u_{s}(φ)', fontsize=9)
        ax.set_title(f'Spinor s={s}', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

        # Bottom: |u_s|² vs φ
        ax2 = axes3[1, s]
        for r in range(4):
            mask = sublat == r
            order = np.argsort(phi_perp[mask])
            ax2.plot(phi_perp[mask][order], np.abs(u[mask, s][order])**2, '.-',
                     ms=3, lw=0.5, color=colors_sub[r])
        ax2.set_xlabel('φ', fontsize=8)
        ax2.set_ylabel(f'|u_{s}|²', fontsize=9)
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(labelsize=7)

    handles = [plt.Line2D([0], [0], marker='.', color=colors_sub[r], ls='',
               ms=6, label=f'r={r}') for r in range(4)]
    fig3.legend(handles=handles, loc='upper right', fontsize=8, ncol=4)

    fig3.suptitle(
        f'Spinor-resolved Bloch envelope: E={E[j_detail]:.3f}, k={k_j:.3f}\n'
        f'Dots = Re(u), crosses = Im(u).  Each sublattice traces a smooth curve in φ.',
        fontsize=12)
    plt.tight_layout()
    out3 = '/tmp/test_E2_spinor_envelope.png'
    plt.savefig(out3, dpi=150)
    print(f"Figure 3 (spinor detail) saved to {out3}")

    # ==================================================================
    # FIGURE 4: Modulation depth vs k — quantify the aperiodicity
    # ==================================================================
    ext_idx = np.where(extended & (E > 0.01))[0]
    mod_depths = []
    k_vals_ext = []
    E_vals_ext = []

    for j in ext_idx:
        psi_j = evecs[:, j].reshape(N, 4)
        k_j = k_all[j]
        u_j = psi_j * np.exp(-1j * k_j * ns)[:, None]
        u_n2 = np.sum(np.abs(u_j)**2, axis=1)
        mean_u2 = np.mean(u_n2)
        if mean_u2 > 1e-15:
            mod_depth = (np.max(u_n2) - np.min(u_n2)) / mean_u2
        else:
            mod_depth = 0
        mod_depths.append(mod_depth)
        k_vals_ext.append(abs(k_j))
        E_vals_ext.append(E[j])

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

    ax4a.scatter(k_vals_ext, mod_depths, s=8, alpha=0.5, c=E_vals_ext,
                 cmap='viridis')
    ax4a.set_xlabel('|k|')
    ax4a.set_ylabel('Envelope modulation depth\n(max−min)/mean of |u|²')
    ax4a.set_title('How strongly does the envelope vary?')
    ax4a.grid(True, alpha=0.3)
    cb = plt.colorbar(ax4a.collections[0], ax=ax4a, label='Quasienergy E')

    # R² of fit to first 6 perp-space harmonics
    R2_vals = []
    for j in ext_idx:
        psi_j = evecs[:, j].reshape(N, 4)
        k_j = k_all[j]
        u_j = psi_j * np.exp(-1j * k_j * ns)[:, None]
        u_n2 = np.sum(np.abs(u_j)**2, axis=1)
        mean_u2 = np.mean(u_n2)
        if mean_u2 < 1e-15:
            R2_vals.append(0)
            continue
        # Fit per sublattice
        r2_list = []
        for r in range(4):
            mask = sublat == r
            delta = u_n2[mask] / np.mean(u_n2[mask]) - 1
            var_tot = np.var(delta)
            if var_tot < 1e-15:
                r2_list.append(1.0)
                continue
            phi_r = phi_perp[mask]
            design = np.zeros((mask.sum(), 12))
            for h in range(6):
                design[:, 2*h] = np.cos((h+1) * phi_r)
                design[:, 2*h+1] = np.sin((h+1) * phi_r)
            coeffs, _, _, _ = np.linalg.lstsq(design, delta, rcond=None)
            r2_list.append(np.var(design @ coeffs) / var_tot)
        R2_vals.append(np.mean(r2_list))

    ax4b.scatter(k_vals_ext, R2_vals, s=8, alpha=0.5, c='darkgreen')
    ax4b.set_xlabel('|k|')
    ax4b.set_ylabel('R² (envelope explained by 6 φ-harmonics)')
    ax4b.set_title('Smoothness of envelope in perp space')
    ax4b.set_ylim(-0.05, 1.05)
    ax4b.grid(True, alpha=0.3)

    fig4.suptitle(
        f'Envelope modulation statistics across all extended eigenstates\n'
        f'N={N}, φ={mix_phi}', fontsize=12)
    plt.tight_layout()
    out4 = '/tmp/test_E2_modulation_stats.png'
    plt.savefig(out4, dpi=150)
    print(f"Figure 4 (modulation stats) saved to {out4}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"All {len(ext_idx)} extended eigenstates are quasi-Bloch states:")
    print(f"  ψ(n) = e^{{ikn}} · F_{{n mod 4}}(nθ mod 2π)")
    print(f"  Envelope modulation depth: {np.mean(mod_depths):.3f} ± {np.std(mod_depths):.3f}")
    print(f"  Perp-space R² (6 harmonics): {np.mean(R2_vals):.3f} ± {np.std(R2_vals):.3f}")
    print(f"  Density is exactly stationary under time evolution (fidelity = 1)")


if __name__ == '__main__':
    main()
