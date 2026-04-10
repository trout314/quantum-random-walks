#!/usr/bin/env python3
"""
L1 density comparison: walk vs Dirac with different ICs.

Compares the walk density evolution against the continuum Dirac solver
using three IC strategies:
  A) Dirac-frame: Gaussian × χ_Dirac (baseline)
  B) Beta-symmetric: Gaussian × (1,0,1,0)/√2 (previous best)
  C) Quasi-Bloch: Gaussian × dressed W(k=0) eigenvector

All compared against Dirac evolution of (1,0,1,0)/√2 Gaussian.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reference'))
from dirac_1d_4comp import solve_dirac_1d_4comp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.helix_geometry import build_taus

BETA = np.diag([1, 1, -1, -1]).astype(complex)

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

def build_chain_taus(N, pat=None):
    """Build tau operators for N sites. Delegates to helix_geometry.build_taus."""
    return build_taus(N)

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


# ---- Quasi-Bloch IC ----

def compute_walk_blocks(N, taus, mix_phi):
    F = np.zeros((N, 4, 4), dtype=complex)
    G = np.zeros((N, 4, 4), dtype=complex)
    for m in range(N):
        V_m = build_vmix_block(taus[m], mix_phi)
        m_prev = (m - 1) % N
        m_next = (m + 1) % N
        U_fwd = frame_transport(taus[m_prev], taus[m])
        U_bwd = frame_transport(taus[m_next], taus[m])
        F[m] = V_m @ U_fwd @ proj_plus(taus[m_prev])
        G[m] = V_m @ U_bwd @ proj_minus(taus[m_next])
    return F, G

def fourier_coefficients_1d(blocks, N, M):
    B_hat = np.zeros((2 * M + 1, 4, 4), dtype=complex)
    ns = np.arange(N)
    for l_idx, l in enumerate(range(-M, M + 1)):
        exp_f = np.exp(-1j * l * ns * THETA_BC)
        for i in range(4):
            for j in range(4):
                B_hat[l_idx, i, j] = np.mean(blocks[:, i, j] * exp_f)
    return B_hat

def build_Wk(k, F_hat, G_hat, M):
    dim = 4 * (2 * M + 1)
    W = np.zeros((dim, dim), dtype=complex)
    for mp in range(-M, M + 1):
        for m in range(-M, M + 1):
            l_idx = (mp - m) + M
            if 0 <= l_idx < 2 * M + 1:
                row = (mp + M) * 4
                col = (m + M) * 4
                W[row:row+4, col:col+4] += (
                    np.exp(-1j * k) * F_hat[l_idx] +
                    np.exp(1j * k) * G_hat[l_idx])
    return W

def make_bloch_ic(N, taus, mix_phi, sigma, M_harm=3):
    """Construct quasi-Bloch IC: Gaussian × W(k=0) eigenvector with dressing."""
    F_bl, G_bl = compute_walk_blocks(N, taus, mix_phi)
    F_hat = fourier_coefficients_1d(F_bl, N, M_harm)
    G_hat = fourier_coefficients_1d(G_bl, N, M_harm)

    Wk0 = build_Wk(0, F_hat, G_hat, M_harm)
    evals, evecs = np.linalg.eig(Wk0)
    E = np.angle(evals)

    # Find positive-energy state nearest the expected mass gap
    m_eff = 0.9 * mix_phi
    pos_mask = E > 0.001
    if pos_mask.any():
        diffs = np.abs(E[pos_mask] - m_eff)
        best = np.where(pos_mask)[0][np.argmin(diffs)]
    else:
        best = np.argmin(np.abs(E - m_eff))

    evec0 = evecs[:, best]
    print(f"  Bloch IC: E(k=0) = {E[best]:.6f}, expected ~{m_eff:.6f}")

    center = N // 2
    xs = np.arange(N) - center
    gauss = np.exp(-xs**2 / (2 * sigma**2))

    psi = np.zeros(4 * N, dtype=complex)
    for n in range(N):
        phi_n = n * THETA_BC
        spinor = np.zeros(4, dtype=complex)
        for m_h in range(-M_harm, M_harm + 1):
            start = (m_h + M_harm) * 4
            spinor += evec0[start:start+4] * np.exp(1j * m_h * phi_n)
        psi[4*n:4*n+4] = gauss[n] * spinor
    psi /= np.linalg.norm(psi)
    return psi


# ---- IC construction ----

def make_dirac_ic(N, chi, sigma):
    """Gaussian × fixed spinor, no frame transport."""
    center = N // 2
    xs = np.arange(N) - center
    gauss = np.exp(-xs**2 / (2 * sigma**2))
    psi = np.zeros(4 * N, dtype=complex)
    for n in range(N):
        psi[4*n:4*n+4] = gauss[n] * chi
    psi /= np.linalg.norm(psi)
    return psi


def make_transported_ic(N, taus, chi, sigma):
    """Gaussian × frame-transported spinor (like the D code)."""
    center = N // 2
    xs = np.arange(N) - center
    gauss = np.exp(-xs**2 / (2 * sigma**2))

    psi = np.zeros(4 * N, dtype=complex)
    # Forward pass from center
    spinor = chi.copy()
    psi[4*center:4*center+4] = gauss[center] * spinor
    for n in range(center + 1, N):
        U = frame_transport(taus[n-1], taus[n])
        spinor = U @ spinor
        spinor /= np.linalg.norm(spinor)
        psi[4*n:4*n+4] = gauss[n] * spinor
    # Backward pass from center
    spinor = chi.copy()
    for n in range(center - 1, -1, -1):
        U = frame_transport(taus[n+1], taus[n])
        spinor = U @ spinor
        spinor /= np.linalg.norm(spinor)
        psi[4*n:4*n+4] = gauss[n] * spinor
    psi /= np.linalg.norm(psi)
    return psi


# ---- Metrics ----

def compute_density(psi, N):
    return np.array([np.sum(np.abs(psi[4*n:4*n+4])**2) for n in range(N)])

def l1_distance(rho1, rho2):
    """L1 distance between two normalized densities."""
    r1 = rho1 / np.sum(rho1)
    r2 = rho2 / np.sum(rho2)
    return np.sum(np.abs(r1 - r2))

def mean_position(rho, xs):
    rho_n = rho / np.sum(rho)
    return np.sum(rho_n * xs)


# ---- Main ----

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    pat = [1, 3, 0, 2]
    m_dirac = mix_phi  # Dirac solver uses m = mix_phi

    print(f"N={N}, φ_mix={mix_phi}, σ={sigma}, steps={n_steps}")
    print(f"Dirac mass: m={m_dirac}")

    print("\nBuilding chain...")
    taus = build_chain_taus(N, pat)

    print("Building walk operator...")
    W = build_full_walk(N, taus, mix_phi)
    unitarity = np.linalg.norm(W.conj().T @ W - np.eye(4*N))
    print(f"Unitarity error: {unitarity:.2e}")

    center = N // 2
    xs = np.arange(N) - center

    # ---- Construct ICs ----
    chi_beta_sym = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

    print("\nConstructing ICs:")
    print("  A) Beta-symmetric (1,0,1,0)/√2, no transport")
    psi_A = make_dirac_ic(N, chi_beta_sym, sigma)

    print("  B) Beta-symmetric, frame-transported")
    psi_B = make_transported_ic(N, taus, chi_beta_sym, sigma)

    print("  C) Quasi-Bloch dressed")
    psi_C = make_bloch_ic(N, taus, mix_phi, sigma, M_harm=3)

    # P+/P- symmetric at origin
    tau0 = taus[center]
    Pp0 = proj_plus(tau0)
    Pm0 = proj_minus(tau0)
    chi_ref = np.array([1, 0, 0, 0], dtype=complex)
    pp = Pp0 @ chi_ref; pp /= np.linalg.norm(pp)
    pm = Pm0 @ chi_ref; pm /= np.linalg.norm(pm)
    chi_pp_pm = (pp + pm) / np.sqrt(2)
    print("  D) P+/P- symmetric, frame-transported")
    psi_D = make_transported_ic(N, taus, chi_pp_pm, sigma)

    ics = [
        ('A: β-sym untransported', psi_A),
        ('B: β-sym transported', psi_B),
        ('C: quasi-Bloch', psi_C),
        ('D: P±-sym transported', psi_D),
    ]

    # ---- Dirac reference ----
    print("\nSolving Dirac equation...")
    N_dirac = 2048
    x_d, times_d, dens_d = solve_dirac_1d_4comp(
        N_dirac, chi_beta_sym, sigma, c=1.0, m=m_dirac, t_max=n_steps, dt_output=10
    )
    rho_dirac_final = dens_d[-1]['total']
    rho_dirac_final /= np.sum(rho_dirac_final)

    # ---- Evolve and measure ----
    print(f"\nEvolving {n_steps} steps...")
    results = {}

    for label, psi0 in ics:
        psi = psi0.copy()
        drift_history = []
        l1_history = []

        for t in range(n_steps + 1):
            rho_walk = compute_density(psi, N)
            xm = mean_position(rho_walk, xs)
            drift_history.append(xm)

            # L1 at output times
            if t % 10 == 0:
                idx_t = t // 10
                if idx_t < len(dens_d):
                    rho_d = dens_d[idx_t]['total']
                    # Interpolate Dirac density to walk grid
                    rho_d_interp = np.interp(xs, x_d, rho_d, left=0, right=0)
                    l1 = l1_distance(rho_walk, rho_d_interp)
                    l1_history.append((t, l1))

            if t < n_steps:
                psi = W @ psi

        rho_final = compute_density(psi, N)  # after last step was applied
        # but we already recorded metrics at t=n_steps in the loop

        results[label] = {
            'drift': drift_history,
            'l1_history': l1_history,
            'rho_final': rho_final,
        }

        final_l1 = l1_history[-1][1] if l1_history else float('nan')
        final_drift = drift_history[-1] - drift_history[0]
        print(f"  {label}: L1={final_l1:.4f}  drift={final_drift:+.2f}")

    # ============================================================
    # Plots
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['C0', 'C1', 'C2', 'C3']

    # Panel 1: L1 vs time
    ax = axes[0, 0]
    for i, (label, res) in enumerate(results.items()):
        ts = [h[0] for h in res['l1_history']]
        l1s = [h[1] for h in res['l1_history']]
        ax.plot(ts, l1s, 'o-', color=colors[i], ms=3, label=label)
    ax.set_xlabel('Time step')
    ax.set_ylabel('L1 distance (walk vs Dirac)')
    ax.set_title('L1 density distance over time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Drift vs time
    ax = axes[0, 1]
    for i, (label, res) in enumerate(results.items()):
        drift = np.array(res['drift']) - res['drift'][0]
        ax.plot(drift, color=colors[i], label=label)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Drift (sites)')
    ax.set_title('Center of mass drift')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Final density comparison
    ax = axes[1, 0]
    ax.plot(x_d, rho_dirac_final, 'k-', lw=2, label='Dirac', zorder=10)
    for i, (label, res) in enumerate(results.items()):
        rho = res['rho_final'] / np.sum(res['rho_final'])
        ax.plot(xs, rho, color=colors[i], alpha=0.7, label=label)
    ax.set_xlabel('Site')
    ax.set_ylabel('P(x)')
    ax.set_title(f'Density at t={n_steps}')
    ax.set_xlim(-4*sigma, 4*sigma)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: Final L1 bar chart
    ax = axes[1, 1]
    labels_short = []
    l1_finals = []
    for label, res in results.items():
        labels_short.append(label.split(':')[0])
        l1_finals.append(res['l1_history'][-1][1])
    bars = ax.bar(range(len(l1_finals)), l1_finals, color=colors[:len(l1_finals)])
    ax.set_xticks(range(len(labels_short)))
    ax.set_xticklabels(labels_short, fontsize=9)
    ax.set_ylabel('L1 distance')
    ax.set_title(f'Final L1 at t={n_steps}')
    for bar, val in zip(bars, l1_finals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Walk vs Dirac L1 comparison: N={N}, φ_mix={mix_phi}, σ={sigma}',
                 fontsize=14)
    plt.tight_layout()

    out = '/tmp/quasi_bloch_l1.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    main()
