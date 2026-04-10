#!/usr/bin/env python3
"""
Dispersion relation via momentum-kick sweep.

Launches wavepackets at various k0, measures group velocity v_g = d<x>/dt,
and compares to the Dirac prediction v_g(k) = k / sqrt(k^2 + m^2).

Usage: python3 scripts/dispersion_k0_sweep.py [sigma] [nsteps] [phi]
Defaults: sigma=30, nsteps=60, phi=0.03 (C=0.9)
"""
import subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WALK_BIN = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'dlang', 'build', 'walk_1d_d')


def run_walk(sigma, mix_phi, nsteps, k0, ic_type=2):
    cmd = [WALK_BIN, '0', str(sigma), str(nsteps),
           '3', '0', str(k0), str(mix_phi), '0', str(ic_type)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    ts, xs, x2s = [], [], []
    for line in result.stdout.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            ts.append(int(parts[0]))
            xs.append(float(parts[2]))
            x2s.append(float(parts[3]))
    return np.array(ts, dtype=float), np.array(xs), np.array(x2s)


def measure_group_velocity(ts, xs):
    """Linear fit of <x>(t) over second half to get v_g."""
    n = len(ts)
    start = n // 2
    if start < 2:
        start = 1
    t_fit = ts[start:]
    x_fit = xs[start:]
    if len(t_fit) < 2:
        return 0.0
    coeffs = np.polyfit(t_fit, x_fit, 1)
    return coeffs[0]


def main():
    sigma = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    phi = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03

    C = phi * sigma
    m = 0.9 * phi  # effective mass

    k0_values = np.linspace(0, 2.5, 25)

    print(f"k0 sweep: sigma={sigma:.0f}, nsteps={nsteps}, phi={phi:.4f}, C={C:.2f}")
    print(f"Dirac mass: m = 0.9*phi = {m:.4f}")
    print(f"{'k0':>8s}  {'v_g':>10s}  {'v_Dirac':>10s}  {'err%':>8s}")

    vgs = []
    for k0 in k0_values:
        ts, xs, x2s = run_walk(sigma, phi, nsteps, k0)
        vg = measure_group_velocity(ts, xs)
        vg_dirac = k0 / np.sqrt(k0**2 + m**2) if m > 0 else (1.0 if k0 > 0 else 0.0)
        err = 100 * (vg - vg_dirac) / vg_dirac if abs(vg_dirac) > 0.01 else 0
        vgs.append(vg)
        print(f"{k0:8.3f}  {vg:10.4f}  {vg_dirac:10.4f}  {err:+7.1f}%")
        sys.stderr.write(f"  k0={k0:.3f} vg={vg:.4f}\n")
        sys.stderr.flush()

    vgs = np.array(vgs)

    # Also integrate v_g to get E(k): E = integral v_g dk
    dk = k0_values[1] - k0_values[0]
    E_walk = np.cumsum(vgs) * dk
    # Shift so E(0) = m (the gap)
    E_walk += m - E_walk[0] if len(E_walk) > 0 else 0

    k_th = np.linspace(0, 2.5, 200)
    vg_dirac = k_th / np.sqrt(k_th**2 + m**2)
    E_dirac = np.sqrt(k_th**2 + m**2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: v_g(k)
    ax = axes[0]
    ax.plot(k0_values, vgs, 'bo-', markersize=5, label='walk')
    ax.plot(k_th, vg_dirac, 'r-', linewidth=2, alpha=0.7,
            label=f'Dirac: k/√(k²+m²), m={m:.4f}')
    ax.set_xlabel('k₀')
    ax.set_ylabel('v_g = d⟨x⟩/dt')
    ax.set_title('Group velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.3)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)

    # Panel 2: E(k) from integrated v_g
    ax = axes[1]
    ax.plot(k0_values, E_walk, 'bo-', markersize=5, label='walk (integrated v_g)')
    ax.plot(k_th, E_dirac, 'r-', linewidth=2, alpha=0.7,
            label=f'Dirac: √(k²+m²), m={m:.4f}')
    ax.plot(k_th, k_th, 'k--', alpha=0.3, label='E=k (massless)')
    ax.set_xlabel('k₀')
    ax.set_ylabel('E(k)')
    ax.set_title('Dispersion relation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Dispersion via k₀ sweep:  σ={sigma:.0f}, nsteps={nsteps}, '
                 f'φ={phi:.4f}, C={C:.2f}', fontsize=14)
    plt.tight_layout()

    outpath = '/tmp/dispersion_k0_sweep.png'
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")


if __name__ == '__main__':
    main()
