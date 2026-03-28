#!/usr/bin/env python3
"""
Publication figure: three independent measurements of the Dirac dispersion.

(a) Eigenvalue spectrum E(k) from walk operator diagonalization
(b) Group velocity from peak-splitting dynamics vs Dirac solver
(c) Wavepacket spreading σ(t) vs Dirac prediction

All three are consistent with E = √(k² + m²), m = 0.878 × φ_mix.
"""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dirac_1d_4comp import solve_dirac_1d_4comp
from quasi_bloch_l1 import (build_chain_taus, build_full_walk,
    make_transported_ic, proj_plus, proj_minus, compute_density, frame_transport)


def make_pp_spinor(taus, center):
    tau0 = taus[center]
    Pp0 = 0.5 * (np.eye(4) + tau0)
    Pm0 = 0.5 * (np.eye(4) - tau0)
    chi_ref = np.array([1, 0, 0, 0], dtype=complex)
    pp = Pp0 @ chi_ref; pp /= np.linalg.norm(pp)
    pm = Pm0 @ chi_ref; pm /= np.linalg.norm(pm)
    return (pp + pm) / np.sqrt(2)


def measure_com_and_width(psi, N):
    center = N // 2
    xs = np.arange(N) - center
    rho = np.array([np.sum(np.abs(psi[4*n:4*n+4])**2) for n in range(N)])
    rho /= np.sum(rho)
    xm = np.sum(rho * xs)
    var = np.sum(rho * (xs - xm)**2)
    return xm, np.sqrt(var)


def main():
    N = 1200; pat = [1, 3, 0, 2]; N_dirac = 4096
    mix_phi = 0.10; mass_ratio = 0.878; m_eff = mix_phi * mass_ratio
    chi_beta = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    center = N // 2; xs = np.arange(N) - center

    print(f"N={N}, φ_mix={mix_phi}, m_eff={m_eff:.4f}")
    print("Building walk operator...")
    taus = build_chain_taus(N, pat)
    W = build_full_walk(N, taus, mix_phi)
    chi_pp = make_pp_spinor(taus, center)

    # ================================================================
    # (a) Eigenvalue spectrum
    # ================================================================
    print("\n--- (a) Eigenvalue spectrum ---")
    # Use smaller N for cleaner spectrum (less crowded)
    N_spec = 600
    taus_spec = build_chain_taus(N_spec, pat)
    W_spec = build_full_walk(N_spec, taus_spec, mix_phi)
    print(f"Diagonalizing {4*N_spec}×{4*N_spec}...")
    evals, evecs = np.linalg.eig(W_spec)
    E_all = np.angle(evals)

    freqs = np.fft.fftfreq(N_spec) * 2 * np.pi
    k_all = np.zeros(len(E_all))
    ipr_all = np.zeros(len(E_all))
    for j in range(len(E_all)):
        psi_j = evecs[:, j].reshape(N_spec, 4)
        total_power = np.zeros(N_spec)
        for a in range(4):
            ft = np.fft.fft(psi_j[:, a])
            total_power += np.abs(ft)**2
        total_power[0] = 0
        k_all[j] = freqs[np.argmax(total_power)]
        prob = np.sum(np.abs(psi_j)**2, axis=1)
        prob /= prob.sum()
        ipr_all[j] = 1.0 / np.sum(prob**2)

    extended = ipr_all > N_spec / 10
    print(f"Extended states: {extended.sum()}/{len(E_all)}")

    # ================================================================
    # (b) Group velocity from peak splitting
    # ================================================================
    print("\n--- (b) Group velocity ---")
    sigmas_vel = [8, 10, 11, 12, 13, 15, 16, 18, 20, 25, 30]
    t_vel = 300

    vel_data = []  # (v_walk, v_dirac)
    for sigma in sigmas_vel:
        # Walk
        psi = make_transported_ic(N, taus, chi_pp, float(sigma))
        for _ in range(t_vel):
            psi = W @ psi
        rho_w = compute_density(psi, N)
        rho_w_sm = gaussian_filter1d(rho_w / np.sum(rho_w), sigma=3)

        # Dirac
        x_d, _, dd = solve_dirac_1d_4comp(N_dirac, chi_beta, float(sigma),
                                            1.0, m_eff, t_vel)
        rho_d = dd[-1]['total']
        rho_d_n = rho_d / np.sum(rho_d)

        # Right peak positions
        thresh = max(sigma * 0.3, 5)
        rmask_w = xs > thresh
        rmask_d = x_d > thresh
        if rmask_w.any() and rmask_d.any():
            xpk_w = xs[rmask_w][np.argmax(rho_w_sm[rmask_w])]
            xpk_d = x_d[rmask_d][np.argmax(rho_d_n[rmask_d])]
            if xpk_w > thresh and xpk_d > thresh:
                v_w = xpk_w / t_vel
                v_d = xpk_d / t_vel
                vel_data.append((v_w, v_d, sigma))
                err = (v_w - v_d) / v_d * 100
                print(f"  σ={sigma:3d}: v_walk={v_w:.4f}, v_dirac={v_d:.4f}, err={err:+.1f}%")
            else:
                print(f"  σ={sigma:3d}: peaks not separated")
        else:
            print(f"  σ={sigma:3d}: peaks not separated")

    # ================================================================
    # (c) Wavepacket spreading
    # ================================================================
    print("\n--- (c) Wavepacket spreading ---")
    sigma_spread = 50.0
    t_max_spread = 600
    t_record = np.arange(0, t_max_spread + 1, 5)

    psi = make_transported_ic(N, taus, chi_pp, sigma_spread)
    width_walk = []
    idx = 0
    for t in range(t_max_spread + 1):
        if idx < len(t_record) and t == t_record[idx]:
            _, w = measure_com_and_width(psi, N)
            width_walk.append(w)
            idx += 1
        if t < t_max_spread:
            psi = W @ psi
    width_walk = np.array(width_walk)

    sigma0 = width_walk[0]
    width_dirac = np.sqrt(sigma0**2 + (t_record / (2 * m_eff * sigma0))**2)
    print(f"  σ₀={sigma0:.2f}, σ({t_max_spread})={width_walk[-1]:.2f}")
    print(f"  Dirac prediction: σ({t_max_spread})={width_dirac[-1]:.2f}")
    resid_spread = (width_walk - width_dirac) / width_dirac * 100
    print(f"  Max residual: {np.max(np.abs(resid_spread)):.1f}%")

    # ================================================================
    # PUBLICATION FIGURE
    # ================================================================
    print("\nGenerating figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- (a) Spectrum ---
    ax = axes[0]
    mask = extended & (np.abs(E_all) > 0.005)
    ax.scatter(np.abs(k_all[mask]), np.abs(E_all[mask]),
               s=3, alpha=0.25, c='#2166ac', rasterized=True, zorder=1)
    k_th = np.linspace(0, 1.2, 300)
    E_dirac = np.sqrt(k_th**2 + m_eff**2)
    ax.plot(k_th, E_dirac, 'r-', lw=2.5, zorder=2,
            label=r'$E = \sqrt{k^2 + m^2}$')
    ax.plot(k_th, k_th, color='gray', ls='--', alpha=0.4, lw=1, label='E = k')
    ax.axhline(m_eff, color='r', ls=':', alpha=0.3)
    ax.set_xlabel('Momentum  k', fontsize=13)
    ax.set_ylabel('Quasi-energy  |E|', fontsize=13)
    ax.set_title('(a)  Eigenvalue spectrum', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.15)
    ax.text(0.08, m_eff + 0.03, f'm = {m_eff:.4f}', fontsize=9, color='red', alpha=0.6)

    # --- (b) Group velocity ---
    ax = axes[1]
    if vel_data:
        v_walks = [d[0] for d in vel_data]
        v_diracs = [d[1] for d in vel_data]
        ax.plot(v_diracs, v_walks, 'o', color='#2166ac', ms=8, mew=1.5, mfc='white',
                zorder=3, label='Walk peak velocity')
    # Perfect agreement line
    vmax = 1.0
    ax.plot([0, vmax], [0, vmax], 'r-', lw=2, label='Perfect agreement', zorder=2)
    ax.set_xlabel('Dirac solver peak velocity', fontsize=13)
    ax.set_ylabel('Walk peak velocity', fontsize=13)
    ax.set_title('(b)  Group velocity', fontsize=14, fontweight='bold')
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    # Residual inset
    if vel_data:
        ax_in = ax.inset_axes([0.52, 0.1, 0.44, 0.38])
        residuals = [(vw - vd) / vd * 100 for vw, vd, _ in vel_data]
        sigs = [d[2] for d in vel_data]
        ax_in.bar(range(len(residuals)), residuals, color='#2166ac', alpha=0.7)
        ax_in.set_xticks(range(len(sigs)))
        ax_in.set_xticklabels([str(s) for s in sigs], fontsize=7, rotation=45)
        ax_in.set_xlabel('σ', fontsize=8)
        ax_in.set_ylabel('Error (%)', fontsize=8)
        ax_in.axhline(0, color='r', lw=1, alpha=0.5)
        ax_in.tick_params(labelsize=7)
        ax_in.set_ylim(-5, 5)
        ax_in.grid(True, alpha=0.2)

    # --- (c) Spreading ---
    ax = axes[2]
    ax.plot(t_record, width_walk, color='#2166ac', lw=1.5, label='Walk', zorder=2)
    ax.plot(t_record, width_dirac, 'r--', lw=2.5,
            label=r'$\sqrt{\sigma_0^2 + (t/2m\sigma_0)^2}$', zorder=3)
    ax.set_xlabel('Time  t', fontsize=13)
    ax.set_ylabel('Wavepacket width  σ(t)', fontsize=13)
    ax.set_title('(c)  Wavepacket spreading', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    # Residual inset
    ax_in2 = ax.inset_axes([0.12, 0.55, 0.45, 0.38])
    ax_in2.plot(t_record, resid_spread, color='#2166ac', lw=1)
    ax_in2.axhline(0, color='r', lw=1, alpha=0.5)
    ax_in2.set_xlabel('t', fontsize=8)
    ax_in2.set_ylabel('Residual (%)', fontsize=8)
    ax_in2.tick_params(labelsize=7)
    ax_in2.grid(True, alpha=0.2)

    fig.suptitle(
        'Three independent confirmations of Dirac dispersion  '
        r'$E = \sqrt{k^2 + m^2}$' + '\n'
        f'Quantum walk on BC helix:  φ_mix = {mix_phi},  '
        f'm = {m_eff:.4f} = {mass_ratio}×φ_mix',
        fontsize=14, y=1.03)
    plt.tight_layout()

    out = '/tmp/dispersion_publication.png'
    plt.savefig(out, dpi=250, bbox_inches='tight', facecolor='white')
    print(f"Saved to {out}")
    out_pdf = '/tmp/dispersion_publication.pdf'
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved to {out_pdf}")


if __name__ == '__main__':
    main()
