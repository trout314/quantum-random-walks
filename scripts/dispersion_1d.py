#!/usr/bin/env python3
"""
1D dispersion relation via spreading speed measurement.

Measures the spreading speed v from x²(t) for the V-mixing walk (no coin).
The dimensionless mass parameter C = φ·σ controls the dispersion:
  v(C) = c² / √(c² + C²),  c = 1.0 site/step

Usage: python3 scripts/dispersion_1d.py [C] [sigmas...]
Defaults: C=0.3, sigma=200,300,400,500
"""
import subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WALK_BIN = os.path.join(os.path.dirname(__file__),
                        '..', 'dlang', 'build', 'walk_1d_d')

def run_walk(sigma, mix_phi, nsteps):
    """Run walk_1d_d with theta=0 (no coin), return (t, x2) arrays."""
    # args: theta sigma nSteps coinType nuType k0 mixPhi
    cmd = [WALK_BIN, '0', str(sigma), str(nsteps),
           '3', '0', '0.0', str(mix_phi)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    ts, x2s = [], []
    for line in result.stdout.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            ts.append(int(parts[0]))
            x2s.append(float(parts[3]))
    return np.array(ts), np.array(x2s)

def measure_spreading_speed(ts, x2s):
    """Fit x²(t) = x²(0) + v²·t² at late times, return v."""
    # Use second half for steady-state ballistic spreading
    n = len(ts)
    start = n // 2
    if start < 3:
        start = 1
    t_fit = ts[start:].astype(float)
    x2_fit = x2s[start:]
    # Fit x² = a + v² * t²
    # Linear regression: x² vs t²
    t2 = t_fit**2
    coeffs = np.polyfit(t2, x2_fit, 1)
    v2 = coeffs[0]
    v = np.sqrt(max(v2, 0))
    return v

def main():
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    if len(sys.argv) > 2:
        sigmas = [float(s) for s in sys.argv[2:]]
    else:
        sigmas = [200, 300, 400, 500]

    # Dirac prediction: v = c² / sqrt(c² + C²), c = 1.0
    c = 1.0
    v_dirac = c**2 / np.sqrt(c**2 + C**2)

    print(f"Spreading speed measurement: C = φσ = {C}")
    print(f"Dirac prediction: v = {v_dirac:.6f}")
    print(f"Sigma values: {sigmas}")
    print()

    results = []
    for sigma in sigmas:
        phi = C / sigma
        # Need enough steps for wavepacket to spread well beyond initial width
        # v ~ 1, so t ~ 3*sigma gives ~3 sigma of spreading
        nsteps = int(3 * sigma)
        sys.stderr.write(f"  sigma={sigma:.0f}, phi={phi:.6f}, nsteps={nsteps} ... ")
        sys.stderr.flush()

        ts, x2s = run_walk(sigma, phi, nsteps)
        v = measure_spreading_speed(ts, x2s)

        sys.stderr.write(f"v={v:.6f}\n")
        results.append((sigma, phi, v, ts, x2s))

    print(f"\n{'sigma':>6s}  {'phi':>10s}  {'C=phi*sigma':>11s}  {'v_spread':>10s}  {'v_Dirac':>10s}  {'err%':>8s}")
    for sigma, phi, v, _, _ in results:
        err = 100 * (v - v_dirac) / v_dirac
        print(f"{sigma:6.0f}  {phi:10.6f}  {phi*sigma:11.4f}  {v:10.6f}  {v_dirac:10.6f}  {err:+7.2f}%")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: x² vs t for each sigma
    ax = axes[0]
    for sigma, phi, v, ts, x2s in results:
        # Subtract initial x² and normalize by t² to show v²
        x2_0 = x2s[0]
        ax.plot(ts, x2s - x2_0, '-', label=f'σ={sigma:.0f}', linewidth=1.5)
    # Overlay Dirac prediction: Δx² = v²·t²
    t_max = max(r[3][-1] for r in results)
    t_th = np.linspace(0, t_max, 200)
    ax.plot(t_th, v_dirac**2 * t_th**2, 'k--', alpha=0.5,
            label=f'Dirac: v={v_dirac:.3f}')
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('$\\Delta x^2 = x^2(t) - x^2(0)$')
    ax.set_title('Wavepacket spreading')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: v vs sigma (should be flat)
    ax = axes[1]
    sigma_arr = [r[0] for r in results]
    v_arr = [r[2] for r in results]
    ax.plot(sigma_arr, v_arr, 'bo-', markersize=8, label='measured')
    ax.axhline(v_dirac, color='r', linestyle='--',
               label=f'Dirac: v={v_dirac:.4f}')
    ax.set_xlabel('$\\sigma$')
    ax.set_ylabel('Spreading speed $v$')
    ax.set_title('Scaling check: v should be independent of σ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Set y-axis range to show detail
    v_mean = np.mean(v_arr)
    v_range = max(abs(v_dirac - min(v_arr)), abs(v_dirac - max(v_arr)), 0.02)
    ax.set_ylim(v_dirac - 3*v_range, v_dirac + 3*v_range)

    fig.suptitle(f'1D V-mixing Walk: C = φσ = {C}, m = 0.9φ\n'
                 f'Dirac prediction: v = c²/√(c²+C²) = {v_dirac:.4f}',
                 fontsize=13)
    plt.tight_layout()

    outpath = '/tmp/dispersion_1d.png'
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")

if __name__ == '__main__':
    main()
