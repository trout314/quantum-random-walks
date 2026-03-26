#!/usr/bin/env python3
"""
Find the minimum nsteps and sigma that give accurate dispersion measurements.

The key insight: smaller sigma means fewer steps to reach ballistic regime,
but too small risks lattice artifacts. We sweep both sigma and nsteps/sigma ratio.
"""
import subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WALK_BIN = os.path.join(os.path.dirname(__file__),
                        '..', 'dlang', 'build', 'walk_1d_d')

def run_walk(sigma, mix_phi, nsteps, ic_type=2):
    cmd = [WALK_BIN, '0', str(sigma), str(nsteps),
           '3', '0', '0.0', str(mix_phi), '0', str(ic_type)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
    n = len(ts)
    start = n // 2
    if start < 3:
        start = 1
    t_fit = ts[start:].astype(float)
    x2_fit = x2s[start:]
    t2 = t_fit**2
    coeffs = np.polyfit(t2, x2_fit, 1)
    v2 = coeffs[0]
    return np.sqrt(max(v2, 0))

def dirac_speed(C, c=1.0):
    return c**2 / np.sqrt(c**2 + C**2)

def main():
    sigmas = [30, 50, 75, 100, 150, 200, 300]
    C_values = [0.0, 0.3, 0.5, 1.0, 2.0, 3.0]
    ratios = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

    print("Minimum steps test: sweep sigma AND nsteps/sigma ratio")
    print(f"sigmas: {sigmas}")
    print(f"C values: {C_values}")
    print(f"ratios: {ratios}")
    print()

    # For each (sigma, C), find the minimum ratio that gives < 2% error
    # results[sigma][C][ratio] = v_measured
    results = {}

    for sigma in sigmas:
        results[sigma] = {}
        print(f"\n{'='*70}")
        print(f"sigma = {sigma}")
        print(f"{'='*70}")

        for C in C_values:
            results[sigma][C] = {}
            phi = C / sigma if C > 0 else 0.0
            v_dirac = dirac_speed(C)

            print(f"\n  C={C:.1f} (phi={phi:.6f}, v_Dirac={v_dirac:.4f}):")
            converged_ratio = None

            for r in ratios:
                nsteps = max(int(r * sigma), 20)
                sys.stderr.write(f"  sigma={sigma}, C={C}, r={r} ... ")
                sys.stderr.flush()

                ts, x2s = run_walk(sigma, phi, nsteps)
                if len(ts) < 4:
                    sys.stderr.write("too few data points\n")
                    results[sigma][C][r] = None
                    continue

                v = measure_spreading_speed(ts, x2s)
                err = 100 * (v - v_dirac) / v_dirac if v_dirac > 0.01 else 0

                results[sigma][C][r] = (v, err)
                sys.stderr.write(f"v={v:.4f} err={err:+.1f}%\n")

                marker = " <-- ✓" if abs(err) < 2 and converged_ratio is None else ""
                if abs(err) < 2 and converged_ratio is None:
                    converged_ratio = r
                print(f"    r={r:5.1f} nsteps={nsteps:5d}  v={v:.4f}  err={err:+6.2f}%{marker}")

    # Summary: minimum ratio for < 2% error at each (sigma, C)
    print(f"\n\n{'='*70}")
    print("SUMMARY: minimum nsteps/sigma for < 2% error")
    print(f"{'='*70}")
    header = f"{'sigma':>6s}"
    for C in C_values:
        header += f"  {'C='+str(C):>8s}"
    print(header)

    for sigma in sigmas:
        row = f"{sigma:6d}"
        for C in C_values:
            min_r = None
            for r in ratios:
                res = results[sigma][C].get(r)
                if res is not None:
                    v, err = res
                    if abs(err) < 2:
                        min_r = r
                        break
            if min_r is not None:
                nsteps = int(min_r * sigma)
                row += f"  {min_r:5.1f}({nsteps:4d})"
            else:
                row += f"  {'---':>8s}"
        print(row)

    # Summary: minimum ratio for < 5% error
    print(f"\nSUMMARY: minimum nsteps/sigma for < 5% error")
    header = f"{'sigma':>6s}"
    for C in C_values:
        header += f"  {'C='+str(C):>8s}"
    print(header)

    for sigma in sigmas:
        row = f"{sigma:6d}"
        for C in C_values:
            min_r = None
            for r in ratios:
                res = results[sigma][C].get(r)
                if res is not None:
                    v, err = res
                    if abs(err) < 5:
                        min_r = r
                        break
            if min_r is not None:
                nsteps = int(min_r * sigma)
                row += f"  {min_r:5.1f}({nsteps:4d})"
            else:
                row += f"  {'---':>8s}"
        print(row)

    # Plot: error vs nsteps for selected (sigma, C) combos
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for ci, C in enumerate(C_values):
        ax = axes[ci // 3][ci % 3]
        for sigma in sigmas:
            nsteps_arr = []
            errs = []
            for r in ratios:
                res = results[sigma][C].get(r)
                if res is not None:
                    v, err = res
                    nsteps_arr.append(int(r * sigma))
                    errs.append(err)
            if nsteps_arr:
                ax.plot(nsteps_arr, errs, 'o-', markersize=4, label=f'σ={sigma}')

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axhspan(-2, 2, alpha=0.1, color='green')
        ax.axhspan(-5, 5, alpha=0.05, color='yellow')
        ax.set_xlabel('nsteps')
        ax.set_ylabel('Error vs Dirac (%)')
        ax.set_title(f'C = {C}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-30, 30)

    fig.suptitle('Dispersion accuracy vs nsteps for various σ and C', fontsize=14)
    plt.tight_layout()

    outpath = '/tmp/min_steps_test.png'
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")

if __name__ == '__main__':
    main()
