#!/usr/bin/env python3
"""
Test B: 1D Propagation

Verifies:
  1. Leading edge propagates at c = 1 site/step, independent of mass
  2. Group velocity (peak motion) decreases with mass
  3. Both measured across multiple phi values

Uses a narrow wavepacket (small sigma) so the leading edge separates
quickly from the bulk. Tracks the rightmost site with probability
above a threshold at each time step.

Usage: python3 scripts/test_B_propagation.py
"""
import subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

WALK_BIN = os.path.join(os.path.dirname(__file__),
                        '..', 'dlang', 'build', 'walk_1d_d')


def run_walk_density(sigma, mix_phi, nsteps):
    """Run walk, return (t, x_mean, x2) and final density profile."""
    cmd = [WALK_BIN, '0', str(sigma), str(nsteps),
           '3', '0', '0.0', str(mix_phi), '0', '0']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    ts, xmeans, x2s = [], [], []
    for line in result.stdout.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            ts.append(int(parts[0]))
            xmeans.append(float(parts[2]))
            x2s.append(float(parts[3]))
    # Also read the density file
    density = None
    try:
        d = np.loadtxt('/tmp/walk_1d_density.dat')
        density = d
    except:
        pass
    return np.array(ts), np.array(xmeans), np.array(x2s), density


def run_walk_widths(sigma, mix_phi, nsteps):
    """Run walk, return (t, active_width) to track leading edge."""
    cmd = [WALK_BIN, '0', str(sigma), str(nsteps),
           '3', '0', '0.0', str(mix_phi), '0', '0']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    ts, widths = [], []
    for line in result.stdout.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            ts.append(int(parts[0]))
            widths.append(int(parts[4]))
    return np.array(ts), np.array(widths)


def measure_edge_speed(ts, widths, sigma):
    """Fit leading edge expansion rate from active width vs t.

    active_width(t) ≈ initial_width + 2*c*t (both edges expand at c).
    Fit slope of width vs t at late times.
    """
    n = len(ts)
    start = n // 3
    t_fit = ts[start:].astype(float)
    w_fit = widths[start:].astype(float)
    coeffs = np.polyfit(t_fit, w_fit, 1)
    edge_speed = coeffs[0] / 2  # factor 2 for both edges
    return edge_speed


def measure_spreading_speed(ts, x2s):
    """Fit x²(t) = x²(0) + v²·t² at late times."""
    n = len(ts)
    start = n // 2
    t_fit = ts[start:].astype(float)
    x2_fit = x2s[start:]
    coeffs = np.polyfit(t_fit**2, x2_fit, 1)
    return np.sqrt(max(coeffs[0], 0))


def main():
    sigma = 30  # narrow wavepacket for quick edge separation
    nsteps = 500
    phis = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.20, 0.30]

    print("=" * 65)
    print("Test B: 1D Propagation")
    print("=" * 65)
    print(f"sigma={sigma}, nsteps={nsteps}")
    print()

    results = []
    for phi in phis:
        sys.stderr.write(f"  phi={phi:.3f} ... ")
        sys.stderr.flush()

        ts, xmeans, x2s, density = run_walk_density(sigma, phi, nsteps)
        ts_w, widths = run_walk_widths(sigma, phi, nsteps)

        c_edge = measure_edge_speed(ts_w, widths, sigma)
        v_spread = measure_spreading_speed(ts, x2s)

        sys.stderr.write(f"c_edge={c_edge:.4f}, v_spread={v_spread:.4f}\n")
        results.append((phi, c_edge, v_spread, ts_w, widths, ts, x2s))

    # Print table
    print(f"{'phi':>8s}  {'c_edge':>8s}  {'v_spread':>10s}  {'C=phi*sigma':>11s}")
    for phi, c_edge, v_spread, _, _, _, _ in results:
        print(f"{phi:8.4f}  {c_edge:8.4f}  {v_spread:10.6f}  {phi*sigma:11.2f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Active width vs t for several phi
    ax = axes[0]
    for phi, c_edge, v_spread, ts_w, widths, _, _ in results:
        if phi in [0.0, 0.04, 0.12, 0.30]:
            ax.plot(ts_w, widths, '-', linewidth=1.5,
                    label=f'$\\phi$={phi}')
    # Overlay c=1 prediction: width = initial + 2*t
    t_th = np.linspace(0, nsteps, 200)
    initial_width = results[0][4][0]
    ax.plot(t_th, initial_width + 2*t_th, 'k--', alpha=0.5,
            label='2t + const (c=1)')
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('Active width (sites)')
    ax.set_title('Leading edge expansion')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Edge speed vs phi
    ax = axes[1]
    phi_arr = np.array([r[0] for r in results])
    c_arr = np.array([r[1] for r in results])
    ax.plot(phi_arr, c_arr, 'bo-', markersize=8)
    ax.axhline(1.0, color='r', linestyle='--', label='c=1 (Dirac)')
    ax.set_xlabel('$\\phi$ (V-mixing angle)')
    ax.set_ylabel('Edge speed (sites/step)')
    ax.set_title('Leading edge speed vs mass')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.1)

    # Panel 3: Spreading speed vs phi
    ax = axes[2]
    v_arr = np.array([r[2] for r in results])
    ax.plot(phi_arr, v_arr, 'go-', markersize=8, label='measured')
    # Dirac prediction: v = c²/sqrt(c² + C²) with c=1, C=phi*sigma
    C_arr = phi_arr * sigma
    v_dirac = 1.0 / np.sqrt(1.0 + C_arr**2)
    ax.plot(phi_arr, v_dirac, 'r--', linewidth=2, label='Dirac: $1/\\sqrt{1+C^2}$')
    ax.set_xlabel('$\\phi$ (V-mixing angle)')
    ax.set_ylabel('Spreading speed')
    ax.set_title('Group velocity vs mass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    passed_edge = all(abs(c - 1.0) < 0.05 for c in c_arr)
    fig.suptitle(f'Test B: Propagation (σ={sigma}, {nsteps} steps)\n'
                 f'Edge speed: {"PASS" if passed_edge else "FAIL"} '
                 f'(range {c_arr.min():.3f}–{c_arr.max():.3f})',
                 fontsize=13)
    plt.tight_layout()

    outpath = '/tmp/test_B_propagation.png'
    plt.savefig(outpath, dpi=150)

    print(f"\nEdge speed range: {c_arr.min():.4f} – {c_arr.max():.4f}")
    print(f"RESULT: {'PASS' if passed_edge else 'FAIL'}")
    print(f"\nPlot saved to {outpath}")


if __name__ == '__main__':
    main()
