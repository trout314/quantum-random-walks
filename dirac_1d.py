#!/usr/bin/env python3
"""
Exact 1D Dirac equation solver using FFT.

H = c * k * sigma_x + m * sigma_z

where sigma_x = [[0,1],[1,0]], sigma_z = [[1,0],[0,-1]].

Eigenvalues: E_± = ±sqrt(c²k² + m²)
Eigenvectors at each k are known analytically.

Method:
1. FFT initial condition to k-space
2. Project onto energy eigenstates at each k
3. Multiply by exp(-iE_±t)
4. IFFT back to position space
"""

import numpy as np
import matplotlib.pyplot as plt

sigma_x = np.array([[0,1],[1,0]], dtype=complex)
sigma_z = np.array([[1,0],[0,-1]], dtype=complex)


def solve_dirac_1d(N, chi, sigma, c, m, t_max, dt_output=None):
    """
    Solve 1D Dirac equation exactly via FFT.

    Parameters:
        N: number of spatial points
        chi: 2-component initial spinor (same at every site, modulated by Gaussian)
        sigma: Gaussian width (in site units)
        c: speed of light
        m: mass
        t_max: total evolution time
        dt_output: time step between outputs (None = just return final state)

    Returns:
        x: position array (centered on 0)
        times: list of output times
        rhos: list of probability density arrays rho[t][x]
    """
    x = np.arange(N) - N//2
    dx = 1.0  # lattice spacing

    # Initial condition: Gaussian envelope * spinor chi
    gauss = np.exp(-x**2 / (2*sigma**2))
    # psi[a, x] = chi[a] * gauss[x], a = 0,1
    psi_x = np.zeros((2, N), dtype=complex)
    psi_x[0] = chi[0] * gauss
    psi_x[1] = chi[1] * gauss

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi_x)**2))
    psi_x /= norm

    # FFT to k-space
    psi_k = np.zeros((2, N), dtype=complex)
    psi_k[0] = np.fft.fft(psi_x[0])
    psi_k[1] = np.fft.fft(psi_x[1])

    # k values corresponding to FFT
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # k in radians per site

    # Energy at each k
    E = np.sqrt(c**2 * k**2 + m**2)
    E = np.where(E < 1e-15, 1e-15, E)

    # Eigenvectors of H(k) = c*k*sigma_x + m*sigma_z
    # H = [[m, ck], [ck, -m]]
    # Eigenvalues: +E, -E
    # Eigenvector for +E: proportional to (m+E, ck)
    # Eigenvector for -E: proportional to (-ck, m+E)
    # (these follow from (H-EI)v=0)

    # At each k, diagonalize H(k) = c*k*sigma_x + m*sigma_z numerically
    # to avoid edge cases at k=0 or m=0.
    c_plus = np.zeros(N, dtype=complex)
    c_minus = np.zeros(N, dtype=complex)
    vp = np.zeros((2, N), dtype=complex)  # eigenvector for +E
    vm = np.zeros((2, N), dtype=complex)  # eigenvector for -E

    for ik in range(N):
        H_k = np.array([[m, c*k[ik]], [c*k[ik], -m]], dtype=complex)
        evals, evecs = np.linalg.eigh(H_k)
        # eigh returns eigenvalues in ascending order: -E, +E
        # So evecs[:,0] is for -E, evecs[:,1] is for +E
        vm[:, ik] = evecs[:, 0]
        vp[:, ik] = evecs[:, 1]
        c_plus[ik] = np.conj(vp[0, ik]) * psi_k[0, ik] + np.conj(vp[1, ik]) * psi_k[1, ik]
        c_minus[ik] = np.conj(vm[0, ik]) * psi_k[0, ik] + np.conj(vm[1, ik]) * psi_k[1, ik]

    # Time evolution and output
    if dt_output is None:
        times = [t_max]
    else:
        times = list(np.arange(0, t_max + dt_output/2, dt_output))

    rhos = []
    for t in times:
        # Evolve: c_±(t) = c_±(0) * exp(-i E_± t)
        cp_t = c_plus * np.exp(-1j * E * t)
        cm_t = c_minus * np.exp(+1j * E * t)  # -E for negative energy

        # Reconstruct psi_k(t) = c_+(t)|+> + c_-(t)|->
        psi_k_t = np.zeros((2, N), dtype=complex)
        psi_k_t[0] = cp_t * vp[0] + cm_t * vm[0]
        psi_k_t[1] = cp_t * vp[1] + cm_t * vm[1]

        # IFFT back to position space
        psi_x_t = np.zeros((2, N), dtype=complex)
        psi_x_t[0] = np.fft.ifft(psi_k_t[0])
        psi_x_t[1] = np.fft.ifft(psi_k_t[1])

        # Probability density
        rho = np.abs(psi_x_t[0])**2 + np.abs(psi_x_t[1])**2
        rhos.append(rho)

    return x, times, rhos


def main():
    import subprocess

    # Use large N to avoid FFT wrap-around: need N > 2*(sigma + c*t_max)
    sigma = 200.0
    c = 1.0
    t_max = 2000
    N = 2 * int(4*sigma + c * t_max) + 2000  # very generous margin
    N_walk = N  # same for walk

    print(f"N={N}, sigma={sigma}, t_max={t_max}")

    # Verify norm for massless case
    chi = np.array([1, 0], dtype=complex)
    x, _, rhos = solve_dirac_1d(N, chi, sigma, c, m=0.0, t_max=t_max)
    norm = np.sum(rhos[0])
    print(f"Massless, chi=(1,0): norm={norm:.10f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: IC = (1,0,0,0) in walk, (1,0) in Dirac
    for col, (theta, m_val) in enumerate([(0.0, 0.0), (0.02, 0.0013), (0.05, 0.00325)]):
        subprocess.run(
            ['./walk_1d', str(N_walk), str(theta), str(sigma), str(int(t_max)), '0'],
            capture_output=True)
        data = np.loadtxt('/tmp/walk_1d_density.dat')
        site_w = data[:, 0]; prob_w = data[:, 2]
        last = subprocess.run(
            ['./walk_1d', str(N_walk), str(theta), str(sigma), str(int(t_max)), '0'],
            capture_output=True, text=True).stdout.strip().split('\n')[-1].split()
        x_mean = float(last[2])

        x, _, rhos = solve_dirac_1d(N, np.array([1, 0], dtype=complex), sigma, c, m_val, t_max)
        rho_d = rhos[0] / np.sum(rhos[0])

        ax = axes[0, col]
        ax.plot(site_w, prob_w, 'b-', linewidth=0.3, label='Walk (1,0,0,0)')
        ax.plot(x, rho_d, 'r--', linewidth=1.5, label=f'Dirac (1,0) m={m_val}')
        ax.set_title(f'θ={theta}, walk xm={x_mean:.0f}')
        ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
        ax.legend(fontsize=9)
        ax.set_xlim(-3000, 3000)

    # Row 2: IC = (1,0,1,0)/sqrt(2) in walk, (1,1)/sqrt(2) in Dirac
    for col, (theta, m_val) in enumerate([(0.0, 0.0), (0.02, 0.0013), (0.05, 0.00325)]):
        subprocess.run(
            ['./walk_1d', str(N_walk), str(theta), str(sigma), str(int(t_max)), '1'],
            capture_output=True)
        data = np.loadtxt('/tmp/walk_1d_density.dat')
        site_w = data[:, 0]; prob_w = data[:, 2]
        last = subprocess.run(
            ['./walk_1d', str(N_walk), str(theta), str(sigma), str(int(t_max)), '1'],
            capture_output=True, text=True).stdout.strip().split('\n')[-1].split()
        x_mean = float(last[2])

        x, _, rhos = solve_dirac_1d(N, np.array([1, 1], dtype=complex)/np.sqrt(2), sigma, c, m_val, t_max)
        rho_d = rhos[0] / np.sum(rhos[0])

        ax = axes[1, col]
        ax.plot(site_w, prob_w, 'b-', linewidth=0.3, label='Walk (1,0,1,0)/√2')
        ax.plot(x, rho_d, 'r--', linewidth=1.5, label=f'Dirac (1,1)/√2 m={m_val}')
        ax.set_title(f'θ={theta}, walk xm={x_mean:.0f}')
        ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
        ax.legend(fontsize=9)
        ax.set_xlim(-3000, 3000)

    plt.suptitle(f'Walk vs FFT Dirac (σ={sigma}, t={t_max}, N={N})', fontsize=14)
    plt.tight_layout()
    plt.savefig('/tmp/walk_1d_fft_dirac.png', dpi=150)
    print(f'Graph saved to: /tmp/walk_1d_fft_dirac.png')


if __name__ == '__main__':
    main()
