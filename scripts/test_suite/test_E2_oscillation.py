#!/usr/bin/env python3
"""
Visualize the real part of a single spinor component of a walk eigenstate
oscillating in time.  Since ψ(t) = e^{-iEt} ψ(0), the real part swings
sinusoidally while the envelope stays fixed.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.helix_geometry import build_taus

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
    freqs = np.fft.fftfreq(N) * 2 * np.pi
    total_power = np.zeros(N)
    for a in range(4):
        ft = np.fft.fft(psi_2d[:, a])
        total_power += np.abs(ft)**2
    total_power[0] = 0
    return freqs[np.argmax(total_power)]


def main():
    N = 200
    mix_phi = 0.08

    print("Building walk operator...")
    taus = build_taus(N)
    W = build_full_walk(N, taus, mix_phi)

    print("Diagonalizing...")
    evals, evecs = np.linalg.eig(W)
    E = np.angle(evals)

    # Extract k and IPR for all states
    ns = np.arange(N)
    freqs = np.fft.fftfreq(N) * 2 * np.pi
    k_all = np.zeros(len(E))
    ipr_all = np.zeros(len(E))
    for j in range(len(E)):
        psi = evecs[:, j].reshape(N, 4)
        prob = np.sum(np.abs(psi)**2, axis=1)
        prob /= prob.sum()
        ipr_all[j] = 1.0 / np.sum(prob**2)
        total_power = np.zeros(N)
        for a in range(4):
            ft = np.fft.fft(psi[:, a])
            total_power += np.abs(ft)**2
        total_power[0] = 0
        k_all[j] = freqs[np.argmax(total_power)]

    # Find eigenstate nearest to target k
    target_k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    # Among positive-energy extended states, find closest |k| to target
    ext_pos = np.where((ipr_all > N/10) & (E > 0.01))[0]
    k_pos = np.abs(k_all[ext_pos])
    j = ext_pos[np.argmin(np.abs(k_pos - target_k))]
    E_j = E[j]
    psi0 = evecs[:, j].reshape(N, 4)
    k_j = extract_k(psi0, N)
    print(f"Found eigenstate: E={E_j:.4f}, k={k_j:.4f}")

    # Pick the spinor component with the largest amplitude
    comp_power = [np.sum(np.abs(psi0[:, s])**2) for s in range(4)]
    s_best = np.argmax(comp_power)
    print(f"Using spinor component s={s_best} (power fractions: "
          f"{[f'{p:.3f}' for p in comp_power]})")

    # Time evolution: ψ(n, t) = e^{-iEt} ψ(n, 0)
    # Re[ψ_s(n, t)] = Re[e^{-iEt}] Re[ψ_s(n,0)] - Im[e^{-iEt}] Im[ψ_s(n,0)]
    # Period T = 2π/E
    T_period = 2 * np.pi / abs(E_j)
    print(f"Oscillation period: T = 2π/|E| = {T_period:.2f} steps")

    psi_s = psi0[:, s_best]  # complex amplitude at t=0

    # Sample times: one full period in ~12 frames
    n_frames = 12
    times = np.linspace(0, T_period, n_frames, endpoint=False)
    # Use fractional time (since the walk is discrete, e^{-iEt} still makes
    # sense as a phase rotation for visualization)

    ymax = 1.3 * np.max(np.abs(psi_s))

    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True, sharey=True)

    for idx, t in enumerate(times):
        ax = axes[idx // 4, idx % 4]
        phase = np.exp(-1j * E_j * t)
        re_psi = np.real(phase * psi_s)

        # Fill positive and negative separately
        pos = np.maximum(re_psi, 0)
        neg = np.minimum(re_psi, 0)
        ax.fill_between(ns, pos, alpha=0.6, color='#2166ac')
        ax.fill_between(ns, neg, alpha=0.6, color='#b2182b')
        ax.plot(ns, re_psi, '-', lw=0.5, color='black')
        ax.axhline(0, color='gray', lw=0.3)

        # Envelope
        ax.plot(ns, np.abs(psi_s), '--', lw=0.6, color='gray', alpha=0.5)
        ax.plot(ns, -np.abs(psi_s), '--', lw=0.6, color='gray', alpha=0.5)

        ax.set_ylim(-ymax, ymax)
        ax.set_xlim(0, N)
        frac = t / T_period
        ax.set_title(f't = {frac:.2f} T', fontsize=11)
        ax.tick_params(labelsize=8)
        if idx // 4 == 2:
            ax.set_xlabel('Site n', fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel(f'Re ψ_{s_best}(n, t)', fontsize=9)

    fig.suptitle(
        f'Eigenstate oscillation:  E = {E_j:.3f},  k = {k_j:.3f},  '
        f'period T = 2π/|E| = {T_period:.1f} steps\n'
        f'Re[ψ_{s_best}(n, t)] — standing wave with quasiperiodic envelope',
        fontsize=13)
    plt.tight_layout()

    out = '/tmp/test_E2_oscillation.png'
    plt.savefig(out, dpi=150)
    print(f"\nSaved to {out}")

    # ==================================================================
    # Figure 2: Standing wave from ψ_k + ψ_{-k} superposition
    # ==================================================================
    # Find the eigenstate with closest k to -k_j (same |E|, opposite momentum)
    target_minus_k = -k_j
    dists = np.abs(k_all[ext_pos] - target_minus_k)
    j_minus = ext_pos[np.argmin(dists)]
    k_minus = k_all[j_minus]
    E_minus = E[j_minus]
    psi_minus = evecs[:, j_minus].reshape(N, 4)
    print(f"\nPartner eigenstate: E={E_minus:.4f}, k={k_minus:.4f}")
    print(f"  |E+ - E-| = {abs(E_j - E_minus):.6f}")
    print(f"  |k+ + k-| = {abs(k_j + k_minus):.6f}")

    dE = E_j - E_minus
    E_avg = (E_j + E_minus) / 2
    if abs(dE) > 1e-10:
        T_beat = 2 * np.pi / abs(dE)
        print(f"  Beat period (E+ vs E-): {T_beat:.1f} steps")
    else:
        T_beat = np.inf

    # Build standing wave: ψ_standing = (ψ_k + ψ_{-k}) / √2
    psi_plus_s = psi0[:, s_best]
    psi_minus_s = psi_minus[:, s_best]
    psi_stand = (psi_plus_s + psi_minus_s) / np.sqrt(2)

    # Traditional standing wave: cos(kn + φ₀) with phase chosen to align
    # peaks with the QB standing wave at t=0.
    # Find the phase offset that maximizes correlation at t=0
    re_qb_0 = np.real(psi_stand)
    best_phi0 = 0
    best_corr = -np.inf
    for trial_phi in np.linspace(0, 2*np.pi, 360, endpoint=False):
        trial = np.cos(k_j * ns + trial_phi)
        corr = np.dot(re_qb_0, trial)
        if corr > best_corr:
            best_corr = corr
            best_phi0 = trial_phi
    psi_stand_trad = np.cos(k_j * ns + best_phi0)
    psi_stand_trad *= np.max(np.abs(psi_stand)) / np.max(np.abs(psi_stand_trad))
    print(f"  Traditional phase offset: φ₀ = {best_phi0:.3f} rad")

    ymax_stand = 1.3 * np.max(np.abs(psi_stand))

    # Time samples: t = 0 to 0.5T in 8 frames
    n_frames_stand = 8
    times_stand = np.linspace(0, 0.5 * T_period, n_frames_stand, endpoint=True)

    fig2, axes2 = plt.subplots(2, n_frames_stand, figsize=(3.5 * n_frames_stand, 7))

    for idx, t in enumerate(times_stand):
        frac = t / T_period

        # Top: quasi-Bloch standing wave
        psi_t = (np.exp(-1j * E_j * t) * psi_plus_s
                 + np.exp(-1j * E_minus * t) * psi_minus_s) / np.sqrt(2)
        re_stand = np.real(psi_t)

        ax = axes2[0, idx]
        ax.fill_between(ns, np.maximum(re_stand, 0), alpha=0.6, color='#2166ac')
        ax.fill_between(ns, np.minimum(re_stand, 0), alpha=0.6, color='#b2182b')
        ax.plot(ns, re_stand, '-', lw=0.5, color='black')
        ax.axhline(0, color='gray', lw=0.3)
        ax.set_ylim(-ymax_stand, ymax_stand)
        ax.set_xlim(0, N)
        ax.set_title(f't = {frac:.2f} T', fontsize=10)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.set_ylabel('QB standing\nψ$_k$+ψ$_{-k}$', fontsize=9,
                           fontweight='bold', color='#2166ac')

        # Bottom: traditional standing wave cos(kn+φ₀) cos(Et)
        ax_tr = axes2[1, idx]
        re_tr = np.cos(E_avg * t) * psi_stand_trad
        ax_tr.fill_between(ns, np.maximum(re_tr, 0), alpha=0.6, color='#2166ac')
        ax_tr.fill_between(ns, np.minimum(re_tr, 0), alpha=0.6, color='#b2182b')
        ax_tr.plot(ns, re_tr, '-', lw=0.5, color='black')
        ax_tr.axhline(0, color='gray', lw=0.3)
        ax_tr.set_ylim(-ymax_stand, ymax_stand)
        ax_tr.set_xlim(0, N)
        ax_tr.set_xlabel('Site n', fontsize=8)
        ax_tr.tick_params(labelsize=7)
        if idx == 0:
            ax_tr.set_ylabel('Traditional\ncos(kn)cos(Et)', fontsize=9,
                              fontweight='bold', color='#666666')

    fig2.suptitle(
        f'Standing wave: (ψ$_k$ + ψ$_{{-k}}$)/√2  vs  cos(kn+φ₀)·cos(Et)\n'
        f'k = {k_j:.3f}, E$_+$ = {E_j:.3f}, E$_-$ = {E_minus:.3f}, '
        f'ΔE = {abs(dE):.4f}'
        + (f', beat period = {T_beat:.0f} steps' if T_beat < 1e6 else ''),
        fontsize=13)
    plt.tight_layout()
    out2 = '/tmp/test_E2_standing.png'
    plt.savefig(out2, dpi=150)
    print(f"Standing wave comparison saved to {out2}")


if __name__ == '__main__':
    main()
