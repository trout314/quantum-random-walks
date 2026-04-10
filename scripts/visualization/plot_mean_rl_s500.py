#!/usr/bin/env python3
"""Run R and L walks at sigma=500, C=0.5, plot mean density vs Dirac."""

import numpy as np
import matplotlib.pyplot as plt
import subprocess, os, sys

sys.path.insert(0, os.path.dirname(__file__))

sigma = 500.0
C = 0.5
phi = C / sigma   # 0.001
theta = 0.0
t_max = 3000      # scale with sigma so wavepacket separates similarly
coin_type = 3
nu_type = 0

c_dirac = 1.0
m_dirac = 0.9 * phi
alpha = np.radians(19.5)
chi = np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)

walk_exe = os.path.join(os.path.dirname(__file__), 'walk_1d')

# Run both spirals
results = {}
for spiral_type, label in [(0, 'R'), (1, 'L')]:
    print(f"Running {label}-spiral (sigma={sigma}, phi={phi:.6f}, t={t_max})...")
    cmd = [walk_exe, str(theta), str(sigma), str(t_max),
           str(coin_type), str(nu_type), '0', str(phi), str(spiral_type)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in proc.stdout.strip().split('\n') if not l.startswith('#')]
    last = lines[-1].split()
    print(f"  {label}: norm={last[1]}, x_mean={last[2]}")
    data = np.loadtxt('/tmp/walk_1d_density.dat')
    results[label] = data

# Mean density
R, L = results['R'], results['L']
site_R, site_L = R[:, 0], L[:, 0]
lo = int(max(site_R[0], site_L[0]))
hi = int(min(site_R[-1], site_L[-1]))
idx_R = (site_R >= lo) & (site_R <= hi)
idx_L = (site_L >= lo) & (site_L <= hi)
site = site_R[idx_R]
prob_mean = 0.5 * (R[idx_R, 2] + L[idx_L, 2])

# Dirac solver
N_dirac = 2 * int(4*sigma + c_dirac * t_max) + 2000
x_d = np.arange(N_dirac) - N_dirac // 2
gauss = np.exp(-x_d**2 / (2*sigma**2))
psi0 = np.zeros((2, N_dirac), dtype=complex)
psi0[0] = chi[0] * gauss; psi0[1] = chi[1] * gauss
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2))

psi_k = np.array([np.fft.fft(psi0[a]) for a in range(2)])
k = np.fft.fftfreq(N_dirac) * 2 * np.pi
E = np.sqrt(c_dirac**2 * k**2 + m_dirac**2)
E = np.where(E < 1e-15, 1e-15, E)

cp, cm = np.zeros(N_dirac, complex), np.zeros(N_dirac, complex)
vp, vm = np.zeros((2, N_dirac), complex), np.zeros((2, N_dirac), complex)
for ik in range(N_dirac):
    H_k = np.array([[m_dirac, c_dirac*k[ik]], [c_dirac*k[ik], -m_dirac]], dtype=complex)
    evals, evecs = np.linalg.eigh(H_k)
    vm[:, ik] = evecs[:, 0]; vp[:, ik] = evecs[:, 1]
    cp[ik] = np.conj(vp[0,ik])*psi_k[0,ik] + np.conj(vp[1,ik])*psi_k[1,ik]
    cm[ik] = np.conj(vm[0,ik])*psi_k[0,ik] + np.conj(vm[1,ik])*psi_k[1,ik]

cp_t = cp * np.exp(-1j * E * t_max)
cm_t = cm * np.exp(+1j * E * t_max)
psi_t = np.array([np.fft.ifft(cp_t*vp[a] + cm_t*vm[a]) for a in range(2)])
rho_tot = np.abs(psi_t[0])**2 + np.abs(psi_t[1])**2
rho_tot /= np.sum(rho_tot)

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(site, prob_mean, 'b-', linewidth=0.6, label='Walk mean(R,L)')
ax.plot(x_d, rho_tot, 'r--', linewidth=1.5, label=f'Dirac (c={c_dirac}, m={m_dirac:.5f})')
ax.set_title(f'Mean(R,L) Walk vs Dirac — C={C}, σ={sigma}, φ={phi:.4f}, t={t_max}', fontsize=13)
ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
ax.legend(fontsize=11)
ax.set_xlim(-4000, 4000)
plt.tight_layout()
out = '/tmp/walk_mean_rl_vs_dirac_s500.png'
plt.savefig(out, dpi=150)
print(f'Graph saved to: {out}')
