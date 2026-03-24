#!/usr/bin/env python3
"""
4-component 1D Dirac solver using the same alpha/beta matrices as the walk.

H = c * k * alpha_1 + m * beta

where alpha_1 and beta are the standard 4x4 Dirac matrices (same as walk_1d.c).
This allows direct comparison with the walk using identical 4-component ICs.

Method: FFT to k-space, diagonalize H(k) at each k, evolve, IFFT back.
"""

import numpy as np

# Standard Dirac matrices (matching walk_1d.c)
alpha_1 = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
], dtype=complex)

alpha_2 = np.array([
    [0, 0, 0, -1j],
    [0, 0, 1j, 0],
    [0, -1j, 0, 0],
    [1j, 0, 0, 0],
], dtype=complex)

alpha_3 = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, -1],
    [1, 0, 0, 0],
    [0, -1, 0, 0],
], dtype=complex)

beta = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)


def solve_dirac_1d_4comp(N, chi, sigma, c, m, t_max, dt_output=None):
    """
    Solve 1D Dirac equation with 4-component spinors via FFT.

    H(k) = c * k * alpha_1 + m * beta

    Parameters:
        N: number of spatial points
        chi: 4-component initial spinor (modulated by Gaussian envelope)
        sigma: Gaussian width (in site units)
        c: speed of light
        m: mass
        t_max: total evolution time
        dt_output: time between outputs (None = final state only)

    Returns:
        x: position array
        times: list of output times
        densities: list of dicts with keys 'total', 'comp' (per-component)
    """
    chi = np.asarray(chi, dtype=complex)
    assert chi.shape == (4,)

    x = np.arange(N) - N // 2

    # Initial condition: Gaussian * chi
    gauss = np.exp(-x**2 / (2 * sigma**2))
    psi_x = np.zeros((4, N), dtype=complex)
    for a in range(4):
        psi_x[a] = chi[a] * gauss

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi_x)**2))
    psi_x /= norm

    # FFT each component
    psi_k = np.zeros((4, N), dtype=complex)
    for a in range(4):
        psi_k[a] = np.fft.fft(psi_x[a])

    # k values
    k = np.fft.fftfreq(N, d=1.0) * 2 * np.pi

    # Diagonalize H(k) at each k
    # Store eigenvalues and eigenvectors
    # evecs_all[ik] is the 4x4 eigenvector matrix at wavenumber ik
    evals_all = np.zeros((N, 4))
    evecs_all = np.zeros((N, 4, 4), dtype=complex)
    coeffs = np.zeros((N, 4), dtype=complex)

    for ik in range(N):
        H_k = c * k[ik] * alpha_1 + m * beta
        evals, evecs = np.linalg.eigh(H_k)
        evals_all[ik] = evals
        evecs_all[ik] = evecs  # evecs[:,j] is j-th eigenvector
        # Project: c_j = <v_j | psi_k>
        for j in range(4):
            coeffs[ik, j] = np.conj(evecs[:, j]) @ psi_k[:, ik]

    # Time evolution
    if dt_output is None:
        times = [t_max]
    else:
        times = list(np.arange(0, t_max + dt_output / 2, dt_output))

    densities = []
    for t in times:
        # Evolve coefficients
        phases = np.exp(-1j * evals_all * t)  # shape (N, 4)

        # Reconstruct psi_k(t)
        psi_k_t = np.zeros((4, N), dtype=complex)
        for j in range(4):
            for a in range(4):
                psi_k_t[a] += coeffs[:, j] * phases[:, j] * evecs_all[:, a, j]

        # IFFT back
        psi_x_t = np.zeros((4, N), dtype=complex)
        for a in range(4):
            psi_x_t[a] = np.fft.ifft(psi_k_t[a])

        # Densities
        comp = np.abs(psi_x_t)**2  # shape (4, N)
        total = np.sum(comp, axis=0)

        # P+ and P- projections (using beta as proxy for tau in continuum limit)
        # P± = (I ± beta)/2
        # |P+ psi|^2 = |psi_0|^2 + |psi_1|^2 (upper two components)
        # |P- psi|^2 = |psi_2|^2 + |psi_3|^2 (lower two components)
        prob_plus = comp[0] + comp[1]
        prob_minus = comp[2] + comp[3]

        densities.append({
            'total': total,
            'comp': comp,
            'prob_plus': prob_plus,
            'prob_minus': prob_minus,
        })

    return x, times, densities


if __name__ == '__main__':
    import subprocess, os, sys

    sigma = 500.0
    C = 0.5
    phi = C / sigma
    theta = 0.0
    t_max = 3000
    coin_type = 3
    nu_type = 0

    c_dirac = 1.0
    m_dirac = 0.9 * phi

    # Use exact same IC as walk: (1, 0, 0, 0)
    chi_4 = np.array([1, 0, 0, 0], dtype=complex)

    N = 2 * int(4 * sigma + c_dirac * t_max) + 2000
    print(f"4-comp Dirac: c={c_dirac}, m={m_dirac:.6f}, sigma={sigma}, t={t_max}, N={N}")
    print(f"IC = {chi_4}")

    # Solve 4-component Dirac
    x_d, times_d, dens_d = solve_dirac_1d_4comp(N, chi_4, sigma, c_dirac, m_dirac, t_max)
    rho_d = dens_d[0]['total']
    rho_d /= np.sum(rho_d)
    pp_d = dens_d[0]['prob_plus'] / np.sum(dens_d[0]['total'])
    pm_d = dens_d[0]['prob_minus'] / np.sum(dens_d[0]['total'])

    # Run walks
    walk_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'walk_1d')
    walk_data = {}
    for spiral_type, label in [(0, 'R'), (1, 'L')]:
        print(f"Running {label}-spiral walk...")
        cmd = [walk_exe, str(theta), str(sigma), str(t_max),
               str(coin_type), str(nu_type), '0', str(phi), str(spiral_type)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines = [l for l in proc.stdout.strip().split('\n') if not l.startswith('#')]
        last = lines[-1].split()
        print(f"  {label}: norm={last[1]}, x_mean={last[2]}")
        walk_data[label] = np.loadtxt('/tmp/walk_1d_density.dat')

    # Mean walk density
    R, L = walk_data['R'], walk_data['L']
    lo = int(max(R[0, 0], L[0, 0]))
    hi = int(min(R[-1, 0], L[-1, 0]))
    idx_R = (R[:, 0] >= lo) & (R[:, 0] <= hi)
    idx_L = (L[:, 0] >= lo) & (L[:, 0] <= hi)
    site = R[idx_R, 0]
    prob_mean = 0.5 * (R[idx_R, 2] + L[idx_L, 2])
    pp_mean = 0.5 * (R[idx_R, 3] + L[idx_L, 3])
    pm_mean = 0.5 * (R[idx_R, 4] + L[idx_L, 4])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(site, prob_mean, 'b-', linewidth=0.6, label='Walk mean(R,L)')
    ax.plot(x_d, rho_d, 'r--', linewidth=1.5, label=f'4-comp Dirac, IC=(1,0,0,0)')
    ax.set_title(f'Mean(R,L) Walk vs 4-comp Dirac — C={C}, σ={sigma}, φ={phi:.4f}, t={t_max}')
    ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
    ax.legend(fontsize=11)
    ax.set_xlim(-4000, 4000)
    plt.tight_layout()
    out = '/tmp/walk_vs_dirac4_s500.png'
    plt.savefig(out, dpi=150)
    print(f'Graph saved to: {out}')
