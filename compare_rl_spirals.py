#!/usr/bin/env python3
"""
Compare R and L spiral walks at C=0.5 against 1D Dirac solver.

Uses V mixing (mix_phi) as mass mechanism with theta=0 (no coin mass).
C = phi * sigma = 0.5

From previous results:
  c = 1.0  (walk speed = 1 site per step)
  m = 0.9 * phi  (mass-phi mapping)
  IC angle alpha = 19.5 deg -> chi = (0.943, 0.334)
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# Import Dirac solver
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dirac_1d import solve_dirac_1d

# Parameters
sigma = 200.0
C = 0.5
phi = C / sigma  # mix_phi = 0.0025
theta = 0.0      # no coin mass, V mixing only
t_max = 1200
coin_type = 3    # dual parity
nu_type = 0      # constant +nu

# Dirac parameters (from previous calibration)
c_dirac = 1.0
m_dirac = 0.9 * phi
alpha_deg = 19.5
alpha = np.radians(alpha_deg)
chi_dirac = np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)

# Dirac solver grid
N_dirac = 2 * int(4*sigma + c_dirac * t_max) + 2000

print(f"Parameters: C={C}, sigma={sigma}, phi={phi:.6f}, theta={theta}")
print(f"Dirac: c={c_dirac}, m={m_dirac:.6f}, alpha={alpha_deg}°, N={N_dirac}")
print(f"Walk: t_max={t_max}, coin={coin_type}")

walk_exe = os.path.join(os.path.dirname(__file__), 'walk_1d')

# Run walks
results = {}
for spiral_type, label in [(0, 'R'), (1, 'L')]:
    print(f"\nRunning {label}-spiral walk...")
    cmd = [walk_exe, str(theta), str(sigma), str(t_max),
           str(coin_type), str(nu_type), '0', str(phi), str(spiral_type)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Parse time evolution from stdout
    lines = [l for l in proc.stdout.strip().split('\n') if not l.startswith('#')]
    last = lines[-1].split()
    print(f"  {label}: norm={last[1]}, x_mean={last[2]}, x2={last[3]}")

    # Read density file
    data = np.loadtxt('/tmp/walk_1d_density.dat')
    results[label] = {
        'site': data[:, 0],
        'pos': data[:, 1],
        'prob': data[:, 2],
        'prob_plus': data[:, 3],
        'prob_minus': data[:, 4],
        'x_mean': float(last[2]),
        'stderr': proc.stderr,
    }
    # Save a copy so R doesn't get overwritten by L
    np.savetxt(f'/tmp/walk_1d_density_{label}.dat', data)

# Solve Dirac
print(f"\nSolving Dirac equation...")
x_d, times_d, rhos_d = solve_dirac_1d(N_dirac, chi_dirac, sigma, c_dirac, m_dirac, t_max)
rho_d = rhos_d[0] / np.sum(rhos_d[0])

# Also get Dirac component densities
# Re-solve to get individual spinor components
x_d2 = np.arange(N_dirac) - N_dirac//2
gauss = np.exp(-x_d2**2 / (2*sigma**2))
psi_x0 = np.zeros((2, N_dirac), dtype=complex)
psi_x0[0] = chi_dirac[0] * gauss
psi_x0[1] = chi_dirac[1] * gauss
norm0 = np.sqrt(np.sum(np.abs(psi_x0)**2))
psi_x0 /= norm0

psi_k = np.zeros((2, N_dirac), dtype=complex)
psi_k[0] = np.fft.fft(psi_x0[0])
psi_k[1] = np.fft.fft(psi_x0[1])

k = np.fft.fftfreq(N_dirac, d=1.0) * 2 * np.pi
E = np.sqrt(c_dirac**2 * k**2 + m_dirac**2)
E = np.where(E < 1e-15, 1e-15, E)

c_plus = np.zeros(N_dirac, dtype=complex)
c_minus = np.zeros(N_dirac, dtype=complex)
vp = np.zeros((2, N_dirac), dtype=complex)
vm = np.zeros((2, N_dirac), dtype=complex)

for ik in range(N_dirac):
    H_k = np.array([[m_dirac, c_dirac*k[ik]], [c_dirac*k[ik], -m_dirac]], dtype=complex)
    evals, evecs = np.linalg.eigh(H_k)
    vm[:, ik] = evecs[:, 0]
    vp[:, ik] = evecs[:, 1]
    c_plus[ik] = np.conj(vp[0, ik]) * psi_k[0, ik] + np.conj(vp[1, ik]) * psi_k[1, ik]
    c_minus[ik] = np.conj(vm[0, ik]) * psi_k[0, ik] + np.conj(vm[1, ik]) * psi_k[1, ik]

cp_t = c_plus * np.exp(-1j * E * t_max)
cm_t = c_minus * np.exp(+1j * E * t_max)
psi_k_t = np.zeros((2, N_dirac), dtype=complex)
psi_k_t[0] = cp_t * vp[0] + cm_t * vm[0]
psi_k_t[1] = cp_t * vp[1] + cm_t * vm[1]
psi_x_t = np.zeros((2, N_dirac), dtype=complex)
psi_x_t[0] = np.fft.ifft(psi_k_t[0])
psi_x_t[1] = np.fft.ifft(psi_k_t[1])

rho_d_total = np.abs(psi_x_t[0])**2 + np.abs(psi_x_t[1])**2
rho_d_1 = np.abs(psi_x_t[0])**2  # component 1 (maps to P+)
rho_d_2 = np.abs(psi_x_t[1])**2  # component 2 (maps to P-)
norm_d = np.sum(rho_d_total)
rho_d_total /= norm_d
rho_d_1 /= norm_d
rho_d_2 /= norm_d

# ---- Plotting ----
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

for col, (spiral, label) in enumerate([(results['R'], 'R-spiral'), (results['L'], 'L-spiral')]):
    site = spiral['site']
    prob = spiral['prob']
    pp = spiral['prob_plus']
    pm = spiral['prob_minus']

    # Row 0: Total probability comparison
    ax = axes[0, col]
    ax.plot(site, prob, 'b-', linewidth=0.5, alpha=0.8, label=f'Walk ({label})')
    ax.plot(x_d2, rho_d_total, 'r--', linewidth=1.5, label=f'Dirac (c={c_dirac}, m={m_dirac:.4f})')
    ax.set_title(f'{label}: Total P(x), C={C}, t={t_max}')
    ax.set_xlabel('Site')
    ax.set_ylabel('P(x)')
    ax.legend(fontsize=9)
    ax.set_xlim(-1500, 1500)

    # Row 1: P+ component comparison
    ax = axes[1, col]
    ax.plot(site, pp, 'b-', linewidth=0.5, alpha=0.8, label=f'Walk P+ ({label})')
    ax.plot(x_d2, rho_d_1, 'r--', linewidth=1.5, label=r'Dirac $|\psi_1|^2$')
    ax.set_title(f'{label}: P+ component')
    ax.set_xlabel('Site')
    ax.set_ylabel('P(x)')
    ax.legend(fontsize=9)
    ax.set_xlim(-1500, 1500)

    # Row 2: P- component comparison
    ax = axes[2, col]
    ax.plot(site, pm, 'b-', linewidth=0.5, alpha=0.8, label=f'Walk P- ({label})')
    ax.plot(x_d2, rho_d_2, 'r--', linewidth=1.5, label=r'Dirac $|\psi_2|^2$')
    ax.set_title(f'{label}: P- component')
    ax.set_xlabel('Site')
    ax.set_ylabel('P(x)')
    ax.legend(fontsize=9)
    ax.set_xlim(-1500, 1500)

plt.suptitle(f'R vs L Spiral Walk vs Dirac — C={C}, σ={sigma}, φ={phi:.4f}, t={t_max}', fontsize=14)
plt.tight_layout()
out_path = '/tmp/walk_rl_spiral_comparison.png'
plt.savefig(out_path, dpi=150)
print(f'\nGraph saved to: {out_path}')

# Also make a direct R vs L overlay
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

ax = axes2[0]
ax.plot(results['R']['site'], results['R']['prob'], 'b-', linewidth=0.5, label='R-spiral')
ax.plot(results['L']['site'], results['L']['prob'], 'r-', linewidth=0.5, label='L-spiral')
ax.set_title(f'R vs L: Total P(x)')
ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
ax.legend(); ax.set_xlim(-1500, 1500)

ax = axes2[1]
ax.plot(results['R']['site'], results['R']['prob_plus'], 'b-', linewidth=0.5, label='R: P+')
ax.plot(results['L']['site'], results['L']['prob_plus'], 'r-', linewidth=0.5, label='L: P+')
ax.set_title('R vs L: P+ component')
ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
ax.legend(); ax.set_xlim(-1500, 1500)

ax = axes2[2]
ax.plot(results['R']['site'], results['R']['prob_minus'], 'b-', linewidth=0.5, label='R: P-')
ax.plot(results['L']['site'], results['L']['prob_minus'], 'r-', linewidth=0.5, label='L: P-')
ax.set_title('R vs L: P- component')
ax.set_xlabel('Site'); ax.set_ylabel('P(x)')
ax.legend(); ax.set_xlim(-1500, 1500)

plt.suptitle(f'R vs L Spiral Direct Comparison — C={C}, σ={sigma}, φ={phi:.4f}, t={t_max}', fontsize=14)
plt.tight_layout()
out_path2 = '/tmp/walk_rl_direct_comparison.png'
plt.savefig(out_path2, dpi=150)
print(f'Graph saved to: {out_path2}')
