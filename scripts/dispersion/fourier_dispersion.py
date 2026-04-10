#!/usr/bin/env python3
"""
Dispersion relation via direct diagonalization of the walk operator.

Builds the full walk operator U on a finite chain (open BCs),
diagonalizes it, and extracts E(k) from eigenvalue phases and
eigenvector Fourier transforms.

No translational symmetry assumed — the BC helix has irrational
screw angle, so there is no unit cell.
"""
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.helix_geometry import build_taus, centroid

# ---- Dirac algebra ----

def proj_plus(tau):
    return 0.5 * (np.eye(4) + tau)

def proj_minus(tau):
    return 0.5 * (np.eye(4) - tau)

def frame_transport(tau_from, tau_to):
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    cos_half = np.sqrt((1 + cos_theta) / 2)
    scale = 1 / (2 * cos_half)
    return scale * (np.eye(4) + prod)

# ---- Build walk operator ----

def build_walk_operator(N, taus, mix_phi, theta=0.0):
    """Build 4N x 4N walk operator matrix.

    Walk step: W = V · S · C
    - C: coin (identity when theta=0)
    - S: shift with frame transport (P+ forward, P- backward)
    - V: post-shift V-mixing
    """
    dim = 4 * N

    # Coin operator (block diagonal)
    C = np.eye(dim, dtype=complex)
    # theta=0 means identity coin, skip

    # Shift operator
    S = np.zeros((dim, dim), dtype=complex)
    for n in range(N):
        Pp = proj_plus(taus[n])
        Pm = proj_minus(taus[n])

        if n + 1 < N:
            # Forward: P+ at site n, transported to site n+1
            U_fwd = frame_transport(taus[n], taus[n+1])
            block = U_fwd @ Pp
            S[4*(n+1):4*(n+1)+4, 4*n:4*n+4] += block

        if n - 1 >= 0:
            # Backward: P- at site n, transported to site n-1
            U_bwd = frame_transport(taus[n], taus[n-1])
            block = U_bwd @ Pm
            S[4*(n-1):4*(n-1)+4, 4*n:4*n+4] += block

    # V-mixing operator (block diagonal)
    V = np.eye(dim, dtype=complex)
    if mix_phi != 0:
        cp, sp = np.cos(mix_phi), np.sin(mix_phi)
        for n in range(N):
            Pp = proj_plus(taus[n])
            Pm = proj_minus(taus[n])

            # Gram-Schmidt for P+ and P- bases
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

            # M = sum_j |pm_j><pp_j| + |pp_j><pm_j|
            M = np.zeros((4, 4), dtype=complex)
            for j in range(2):
                M += np.outer(pm_basis[:, j], pp_basis[:, j].conj())
                M += np.outer(pp_basis[:, j], pm_basis[:, j].conj())

            Vmix = cp * np.eye(4) + 1j * sp * M
            V[4*n:4*n+4, 4*n:4*n+4] = Vmix

    # Full walk operator
    W = V @ S @ C
    return W

# ---- Main ----

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08

    print(f"Building chain: N={N} sites, phi={phi}")
    taus = build_taus(N)

    print(f"Building walk operator: {4*N}x{4*N} matrix")
    W = build_walk_operator(N, taus, phi)

    print("Diagonalizing...")
    eigenvalues, eigenvectors = np.linalg.eig(W)

    # Extract quasienergies from eigenvalue phases
    E = np.angle(eigenvalues)  # E in [-pi, pi]

    # For each eigenvector, compute effective momentum from spatial Fourier transform.
    # The eigenvector has 4N components; reshape to (N, 4) and compute the
    # probability density at each site, then Fourier transform.
    print("Computing effective momenta...")
    k_eff = np.zeros(len(E))
    ipr = np.zeros(len(E))  # inverse participation ratio

    for j in range(len(E)):
        psi = eigenvectors[:, j].reshape(N, 4)
        prob = np.sum(np.abs(psi)**2, axis=1)  # density per site
        prob /= prob.sum()

        # IPR: 1/sum(prob^2), measures localization (N for extended, 1 for localized)
        ipr[j] = 1.0 / np.sum(prob**2)

        # Dominant Fourier component of the spinor amplitude (not density).
        # Sum |FT|^2 over all 4 spinor components for the total spectral weight.
        freqs = np.fft.fftfreq(N) * 2 * np.pi
        total_power = np.zeros(N)
        for a in range(4):
            ft = np.fft.fft(psi[:, a])
            total_power += np.abs(ft)**2
        total_power[0] = 0  # ignore DC
        peak = np.argmax(total_power)
        k_eff[j] = abs(freqs[peak])  # use |k| since E(k)=E(-k)

    # Filter: keep only extended states (IPR > N/10)
    extended = ipr > N / 10
    print(f"Extended states: {extended.sum()} / {len(E)}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: E vs k — full range
    ax = axes[0]
    ax.scatter(k_eff[extended], np.abs(E[extended]), s=2, alpha=0.5, c='blue')
    k_th = np.linspace(0, np.pi, 200)
    m_dirac = 0.9 * phi
    E_dirac = np.sqrt(k_th**2 + m_dirac**2)
    ax.plot(k_th, E_dirac, 'r-', linewidth=2, label=f'Dirac: m=0.9$\\phi$={m_dirac:.4f}')
    ax.plot(k_th, k_th, 'k--', alpha=0.3, label='E=k (massless)')
    ax.set_xlabel('k (effective)')
    ax.set_ylabel('|E|')
    ax.set_title('Dispersion: full range')
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, np.pi)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: E vs k — zoomed to small k
    ax = axes[1]
    zoom = 5 * m_dirac  # zoom range
    ax.scatter(k_eff[extended], np.abs(E[extended]), s=8, alpha=0.5, c='blue')
    k_z = np.linspace(0, zoom, 200)
    ax.plot(k_z, np.sqrt(k_z**2 + m_dirac**2), 'r-', linewidth=2,
            label=f'$\\sqrt{{k^2+m^2}}$, m={m_dirac:.4f}')
    ax.plot(k_z, k_z, 'k--', alpha=0.3, label='E=k')
    ax.axhline(m_dirac, color='r', linestyle=':', alpha=0.5, label=f'm={m_dirac:.4f}')
    ax.set_xlabel('k (effective)')
    ax.set_ylabel('|E|')
    ax.set_title('Dispersion: zoomed near gap')
    ax.set_xlim(0, zoom)
    ax.set_ylim(0, zoom)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Density of states
    ax = axes[2]
    ax.hist(np.abs(E[extended]), bins=50, density=True, alpha=0.7, label='walk DOS')
    ax.axvline(m_dirac, color='r', linestyle='--', label=f'm={m_dirac:.4f}')
    ax.set_xlabel('|E|')
    ax.set_ylabel('DOS')
    ax.set_title('Density of states')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Walk operator spectrum: N={N}, $\\phi$={phi}, C=$\\phi$N={phi*N:.0f}', fontsize=14)
    plt.tight_layout()

    out = '/tmp/fourier_dispersion.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

if __name__ == '__main__':
    main()
