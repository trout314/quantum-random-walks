#!/usr/bin/env python3
"""Plot mean of R and L spiral densities vs Dirac solver."""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from dirac_1d import solve_dirac_1d

# Load saved walk data
R = np.loadtxt('/tmp/walk_1d_density_R.dat')
L = np.loadtxt('/tmp/walk_1d_density_L.dat')

# Both have same site indices (centered on 0)
site_R = R[:, 0]
site_L = L[:, 0]

# Build mean on overlapping range
lo = int(max(site_R[0], site_L[0]))
hi = int(min(site_R[-1], site_L[-1]))

# Index into each array
idx_R = (site_R >= lo) & (site_R <= hi)
idx_L = (site_L >= lo) & (site_L <= hi)

site = site_R[idx_R]
prob_mean = 0.5 * (R[idx_R, 2] + L[idx_L, 2])
pp_mean = 0.5 * (R[idx_R, 3] + L[idx_L, 3])
pm_mean = 0.5 * (R[idx_R, 4] + L[idx_L, 4])

# Dirac solver
sigma = 200.0
phi = 0.0025
c_dirac = 1.0
m_dirac = 0.9 * phi
t_max = 1200
alpha = np.radians(19.5)
chi = np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)
N_dirac = 2 * int(4*sigma + c_dirac * t_max) + 2000

# Full Dirac solve with components
x_d = np.arange(N_dirac) - N_dirac//2
gauss = np.exp(-x_d**2 / (2*sigma**2))
psi0 = np.zeros((2, N_dirac), dtype=complex)
psi0[0] = chi[0] * gauss
psi0[1] = chi[1] * gauss
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
rho1 = np.abs(psi_t[0])**2
rho2 = np.abs(psi_t[1])**2
rho_tot = rho1 + rho2
norm_d = np.sum(rho_tot)
rho_tot /= norm_d; rho1 /= norm_d; rho2 /= norm_d

# ---- Plot ----
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

ax = axes[0]
ax.plot(site, prob_mean, 'b-', linewidth=0.8, label='Walk mean(R,L)')
ax.plot(x_d, rho_tot, 'r--', linewidth=1.5, label=f'Dirac (c={c_dirac}, m={m_dirac:.4f})')
ax.set_title(f'Total P(x) — C=0.5, σ={sigma}, φ={phi}, t={t_max}')
ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
ax.legend(); ax.set_xlim(-1500, 1500)

ax = axes[1]
ax.plot(site, pp_mean, 'b-', linewidth=0.8, label='Walk mean(R,L) P+')
ax.plot(x_d, rho1, 'r--', linewidth=1.5, label=r'Dirac $|\psi_1|^2$')
ax.set_title('P+ component')
ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
ax.legend(); ax.set_xlim(-1500, 1500)

ax = axes[2]
ax.plot(site, pm_mean, 'b-', linewidth=0.8, label='Walk mean(R,L) P-')
ax.plot(x_d, rho2, 'r--', linewidth=1.5, label=r'Dirac $|\psi_2|^2$')
ax.set_title('P- component')
ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
ax.legend(); ax.set_xlim(-1500, 1500)

plt.suptitle(f'Mean(R,L) Walk vs Dirac — C=0.5, σ={sigma}, φ={phi:.4f}, t={t_max}', fontsize=14)
plt.tight_layout()
out = '/tmp/walk_mean_rl_vs_dirac.png'
plt.savefig(out, dpi=150)
print(f'Graph saved to: {out}')
