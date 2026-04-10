#!/usr/bin/env python3
"""
Publication figure: bidirectional walk vs standard walk vs Dirac solver.

Recreates the walk-vs-Dirac density comparison with the layout:
  - Large top panel: final time (t=400)
  - Four small bottom panels: t=0, 100, 200, 300

Compares standard walk (V @ S_exit) and bidirectional walk
(V @ S_entry @ V @ S_exit) against the continuum Dirac equation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys, os

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_root, 'scripts', 'reference'))
from dirac_1d_4comp import solve_dirac_1d_4comp

sys.path.insert(0, _root)
from src.helix_geometry import build_taus, build_entry_taus, make_tau

THETA_BC = np.arccos(-2/3)


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
    pp_b = np.zeros((4, 2), dtype=complex)
    pm_b = np.zeros((4, 2), dtype=complex)
    np_f = nm_f = 0
    for col in range(4):
        if np_f >= 2 and nm_f >= 2: break
        if np_f < 2:
            v = Pp[:, col].copy()
            for j in range(np_f):
                v -= np.vdot(pp_b[:, j], v) * pp_b[:, j]
            nm = np.real(np.vdot(v, v))
            if nm > 1e-10:
                pp_b[:, np_f] = v / np.sqrt(nm); np_f += 1
        if nm_f < 2:
            v = Pm[:, col].copy()
            for j in range(nm_f):
                v -= np.vdot(pm_b[:, j], v) * pm_b[:, j]
            nm = np.real(np.vdot(v, v))
            if nm > 1e-10:
                pm_b[:, nm_f] = v / np.sqrt(nm); nm_f += 1
    M = np.zeros((4, 4), dtype=complex)
    for j in range(2):
        M += np.outer(pm_b[:, j], pp_b[:, j].conj())
        M += np.outer(pp_b[:, j], pm_b[:, j].conj())
    return cp * np.eye(4) + 1j * sp * M


def build_shift(N, taus):
    """Build shift operator S from tau operators."""
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
    return S


def build_vmix(N, taus, mix_phi):
    """Build block-diagonal V-mixing operator."""
    dim = 4 * N
    V = np.eye(dim, dtype=complex)
    if mix_phi != 0:
        for n in range(N):
            V[4*n:4*n+4, 4*n:4*n+4] = build_vmix_block(taus[n], mix_phi)
    return V


def build_standard_walk(N, taus_exit, mix_phi):
    """W = V_exit @ S_exit"""
    S = build_shift(N, taus_exit)
    V = build_vmix(N, taus_exit, mix_phi)
    return V @ S


def build_bidir_walk(N, taus_exit, taus_entry, mix_phi):
    """W = V_exit @ S_entry @ V_exit @ S_exit"""
    S_exit = build_shift(N, taus_exit)
    S_entry = build_shift(N, taus_entry)
    V = build_vmix(N, taus_exit, mix_phi)
    return V @ S_entry @ V @ S_exit


def compute_density(psi, N):
    return np.array([np.sum(np.abs(psi[4*n:4*n+4])**2) for n in range(N)])


def make_transported_ic(N, taus, chi, sigma, k0=0.0):
    """Gaussian wavepacket with frame transport along the chain."""
    center = N // 2
    xs = np.arange(N) - center
    gauss = np.exp(-xs**2 / (2 * sigma**2))
    psi = np.zeros(4 * N, dtype=complex)

    # Forward from center
    chi_cur = chi.copy()
    for n in range(center, N):
        x = n - center
        phase = np.exp(1j * k0 * x)
        psi[4*n:4*n+4] = gauss[n] * phase * chi_cur
        if n < N - 1:
            U = frame_transport(taus[n], taus[n+1])
            chi_cur = U @ chi_cur

    # Backward from center
    chi_cur = chi.copy()
    for n in range(center - 1, -1, -1):
        U = frame_transport(taus[n+1], taus[n])
        chi_cur = U @ chi_cur
        x = n - center
        phase = np.exp(1j * k0 * x)
        psi[4*n:4*n+4] = gauss[n] * phase * chi_cur

    psi /= np.linalg.norm(psi)
    return psi


def make_uniform_ic(N, chi, sigma, k0=0.0):
    """Gaussian wavepacket with uniform spinor (no frame transport)."""
    center = N // 2
    xs = np.arange(N) - center
    gauss = np.exp(-xs**2 / (2 * sigma**2))
    psi = np.zeros(4 * N, dtype=complex)
    for n in range(N):
        x = n - center
        phase = np.exp(1j * k0 * x)
        psi[4*n:4*n+4] = gauss[n] * phase * chi
    psi /= np.linalg.norm(psi)
    return psi


def rms_relative_error(rho_walk, rho_dirac, xs_w, xs_d):
    """RMS relative error between walk and Dirac densities."""
    rho_d_interp = np.interp(xs_w, xs_d, rho_dirac, left=0, right=0)
    mask = rho_d_interp > 1e-6 * np.max(rho_d_interp)
    if not np.any(mask):
        return float('nan')
    rel = (rho_walk[mask] - rho_d_interp[mask]) / rho_d_interp[mask]
    return np.sqrt(np.mean(rel**2)) * 100


def l1_vs_dirac(N, taus_exit, taus_entry, mix_phi_bidir, chi_bidir, sigma,
                dirac_snaps_at, x_d, xs, n_steps, t_eval):
    """Build bidirectional walk with given mix_phi, evolve, return L1 at t_eval."""
    W = build_bidir_walk(N, taus_exit, taus_entry, mix_phi_bidir)
    psi = make_uniform_ic(N, chi_bidir, sigma)
    sm = 3.0
    for t in range(t_eval + 1):
        if t == t_eval:
            rho = compute_density(psi, N)
            rho /= np.sum(rho)
            rho_sm = gaussian_filter1d(rho, sm)
            rho_d = dirac_snaps_at
            rho_d_interp = np.interp(xs, x_d, rho_d, left=0, right=0)
            return np.sum(np.abs(rho_sm - rho_d_interp))
        psi = W @ psi


def main():
    # Parameters matching the old graph
    N = 1200
    sigma = 20.0
    mix_phi_std = 0.10
    mass_ratio = 0.878
    m_eff = mix_phi_std * mass_ratio  # ~0.0878
    n_steps = 400
    snapshot_times = [0, 100, 200, 300, 400]
    N_dirac = 4096

    center = N // 2
    xs = np.arange(N) - center

    # IC: P+/P- symmetric, frame-transported (for standard walk)
    taus_exit = build_taus(N)
    taus_entry = build_entry_taus(N)
    tau0 = taus_exit[center]
    Pp0 = proj_plus(tau0)
    Pm0 = proj_minus(tau0)
    chi_ref = np.array([1, 0, 0, 0], dtype=complex)
    pp = Pp0 @ chi_ref; pp /= np.linalg.norm(pp)
    pm = Pm0 @ chi_ref; pm /= np.linalg.norm(pm)
    chi_ic = (pp + pm) / np.sqrt(2)

    # Beta-symmetric IC for bidirectional walk (no frame transport)
    chi_beta = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

    # Dirac reference
    chi_dirac = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    print("Solving Dirac equation...")
    x_d, times_d, dens_d = solve_dirac_1d_4comp(
        N_dirac, chi_dirac, sigma, c=1.0, m=m_eff, t_max=n_steps, dt_output=1
    )
    dirac_snaps = {}
    for i, t in enumerate(times_d):
        t_int = int(round(t))
        if t_int in set(snapshot_times):
            dirac_snaps[t_int] = dens_d[i]['total'] / np.sum(dens_d[i]['total'])

    # ---- Search for best mix_phi for bidirectional walk ----
    t_eval = 300  # compare at this time
    print(f"\nSearching for best mix_phi (bidirectional) at t={t_eval}...")

    # Coarse sweep
    phi_candidates = np.arange(0.10, 0.80, 0.02)
    best_phi = None
    best_l1 = float('inf')
    for phi in phi_candidates:
        l1 = l1_vs_dirac(N, taus_exit, taus_entry, phi, chi_beta, sigma,
                         dirac_snaps[t_eval], x_d, xs, n_steps, t_eval)
        tag = ""
        if l1 < best_l1:
            best_l1 = l1
            best_phi = phi
            tag = " <-- best"
        print(f"  phi={phi:.3f}: L1={l1:.4f}{tag}")

    # Fine sweep around best
    print(f"\nFine sweep around phi={best_phi:.3f}...")
    phi_fine = np.arange(best_phi - 0.02, best_phi + 0.025, 0.005)
    for phi in phi_fine:
        if phi <= 0:
            continue
        l1 = l1_vs_dirac(N, taus_exit, taus_entry, phi, chi_beta, sigma,
                         dirac_snaps[t_eval], x_d, xs, n_steps, t_eval)
        tag = ""
        if l1 < best_l1:
            best_l1 = l1
            best_phi = phi
            tag = " <-- best"
        print(f"  phi={phi:.4f}: L1={l1:.4f}{tag}")

    mix_phi_bidir = best_phi
    print(f"\nBest bidirectional mix_phi = {mix_phi_bidir:.4f} (L1={best_l1:.4f})")

    # ---- Build final operators ----
    print(f"\nBuilding walk operators (std phi={mix_phi_std}, bidir phi={mix_phi_bidir})...")
    W_std = build_standard_walk(N, taus_exit, mix_phi_std)
    W_bidir = build_bidir_walk(N, taus_exit, taus_entry, mix_phi_bidir)

    psi0_std = make_transported_ic(N, taus_exit, chi_ic, sigma)
    psi0_bidir = make_uniform_ic(N, chi_beta, sigma)

    # Evolve both walks
    print("Evolving walks...")
    psi_std = psi0_std.copy()
    psi_bidir = psi0_bidir.copy()
    std_snaps = {}
    bidir_snaps = {}
    snap_set = set(snapshot_times)

    for t in range(n_steps + 1):
        if t in snap_set:
            rho_std = compute_density(psi_std, N)
            rho_bidir = compute_density(psi_bidir, N)
            std_snaps[t] = rho_std / np.sum(rho_std)
            bidir_snaps[t] = rho_bidir / np.sum(rho_bidir)
            norm_std = np.sqrt(np.sum(np.abs(psi_std)**2))
            norm_bidir = np.sqrt(np.sum(np.abs(psi_bidir)**2))
            print(f"  t={t}: norm_std={norm_std:.8f} norm_bidir={norm_bidir:.8f}")

        if t < n_steps:
            psi_std = W_std @ psi_std
            psi_bidir = W_bidir @ psi_bidir

    # Smoothing width
    sm = 3.0

    # ============================================================
    # Publication figure: big top panel + 4 small bottom panels
    # ============================================================
    fig = plt.figure(figsize=(16, 10))

    # Top panel: t = 400 (final time)
    ax_top = fig.add_axes([0.06, 0.45, 0.90, 0.48])
    t_final = snapshot_times[-1]

    rho_std_sm = gaussian_filter1d(std_snaps[t_final], sm)
    rho_bidir_sm = gaussian_filter1d(bidir_snaps[t_final], sm)

    ax_top.plot(xs, rho_std_sm, '-', color='#2166ac', lw=2,
                label='Standard walk', zorder=5)
    ax_top.plot(xs, rho_bidir_sm, '-', color='#b2182b', lw=2,
                label='Bidirectional walk', zorder=4)
    if t_final in dirac_snaps:
        ax_top.plot(x_d, dirac_snaps[t_final], '--', color='#333333', lw=2,
                    label='1+1D Dirac equation', zorder=10)

    # RMS error annotations
    if t_final in dirac_snaps:
        rms_std = rms_relative_error(rho_std_sm, dirac_snaps[t_final], xs, x_d)
        rms_bidir = rms_relative_error(rho_bidir_sm, dirac_snaps[t_final], xs, x_d)
        ax_top.annotate(f'Standard RMS error: {rms_std:.1f}%',
                       xy=(0.02, 0.92), xycoords='axes fraction',
                       fontsize=10, color='#2166ac',
                       bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    ax_top.set_xlim(-400, 400)
    ax_top.set_ylabel(r'Probability density  ($\times 10^{-3}$)', fontsize=12)
    ax_top.set_xlabel('Position (lattice sites)', fontsize=12)
    ax_top.set_title(f't = {t_final} steps', fontsize=13, fontweight='bold')
    ax_top.legend(fontsize=11, loc='upper right')
    ax_top.grid(True, alpha=0.15)

    # Bottom panels: t = 0, 100, 200, 300
    bottom_times = snapshot_times[:-1]
    n_bottom = len(bottom_times)
    panel_w = 0.90 / n_bottom
    for idx, t in enumerate(bottom_times):
        ax = fig.add_axes([0.06 + idx * panel_w, 0.06, panel_w * 0.92, 0.30])

        rho_std_sm = gaussian_filter1d(std_snaps[t], sm)
        rho_bidir_sm = gaussian_filter1d(bidir_snaps[t], sm)

        ax.plot(xs, rho_std_sm, '-', color='#2166ac', lw=1.5, zorder=5)
        ax.plot(xs, rho_bidir_sm, '-', color='#b2182b', lw=1.5, zorder=4)
        if t in dirac_snaps:
            ax.plot(x_d, dirac_snaps[t], '--', color='#333333', lw=1.5, zorder=10)

        # RMS annotation
        if t in dirac_snaps and t > 0:
            rms_s = rms_relative_error(rho_std_sm, dirac_snaps[t], xs, x_d)
            ax.annotate(f'RMS {rms_s:.1f}%', xy=(0.65, 0.88), xycoords='axes fraction',
                       fontsize=8, color='#2166ac',
                       bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.8))

        ax.set_xlim(-400, 400)
        ax.set_title(f't = {t}', fontsize=11)
        ax.set_xlabel('Position', fontsize=9)
        if idx == 0:
            ax.set_ylabel(r'P(x) ($\times 10^{-3}$)', fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

    fig.suptitle(
        f'Quantum walk on the Boerdijk-Coxeter helix reproduces 1+1D Dirac dynamics\n'
        f'Dirac: m = {m_eff:.4f}, $\\sigma_0$ = {sigma:.0f}    '
        f'Standard: $\\phi_{{mix}}$ = {mix_phi_std}    '
        f'Bidirectional: $\\phi_{{mix}}$ = {mix_phi_bidir:.4f}',
        fontsize=13, y=0.99)

    out = '/tmp/bidir_vs_dirac.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {out}")

    out_pdf = '/tmp/bidir_vs_dirac.pdf'
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"PDF saved to {out_pdf}")


if __name__ == '__main__':
    main()
