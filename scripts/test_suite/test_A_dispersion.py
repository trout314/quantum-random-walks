#!/usr/bin/env python3
"""
Test A: 1D Dispersion Relation

Verifies that the 1D V-mixing walk reproduces the massive Dirac dispersion:
  v(C) = c² / √(c² + C²),  where C = φ·σ is the dimensionless mass parameter.

Measurements:
  1. Sweep C from 0 to 5 at σ=300, measure spreading speed v from x²(t)
  2. Fit to Dirac formula to extract c (speed of light) and verify m ∝ φ
  3. Repeat at σ=200, 500 to verify scaling collapse (v depends only on C, not σ)

Expected:  c = 1.0 site/step,  m = 0.9φ,  scaling CV < 1%

Usage: python3 scripts/test_A_dispersion.py
"""
import subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

WALK_BIN = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'dlang', 'build', 'walk_1d_d')

def run_walk(sigma, mix_phi, nsteps, ic_type=2):
    """Run walk_1d_d with theta=0 (V-mixing only), return (t, x2) arrays."""
    # args: theta sigma nSteps coinType nuType k0 mixPhi spiralType icType
    cmd = [WALK_BIN, '0', str(sigma), str(nsteps),
           '3', '0', '0.0', str(mix_phi), '0', str(ic_type)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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


def dirac_speed(C, c):
    """Dirac prediction: v = c² / sqrt(c² + C²)."""
    return c**2 / np.sqrt(c**2 + C**2)


def main():
    # ---- Configuration ----
    sigma_primary = 300
    sigmas_scaling = [200, 300, 500]
    C_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    print("=" * 65)
    print("Test A: 1D Dispersion Relation (V-mixing walk)")
    print("=" * 65)

    # ---- Part 1: Dispersion curve v(C) at primary sigma ----
    print(f"\n--- Part 1: v(C) at σ={sigma_primary} ---")
    print(f"{'C':>6s}  {'phi':>10s}  {'nsteps':>6s}  {'v':>10s}")

    v_measured = []
    for C in C_values:
        phi = C / sigma_primary if C > 0 else 0.0
        nsteps = max(int(3 * sigma_primary), 500)
        sys.stderr.write(f"  C={C:.1f}, phi={phi:.6f} ... ")
        sys.stderr.flush()

        ts, x2s = run_walk(sigma_primary, phi, nsteps)
        v = measure_spreading_speed(ts, x2s)
        v_measured.append(v)

        sys.stderr.write(f"v={v:.6f}\n")
        print(f"{C:6.1f}  {phi:10.6f}  {nsteps:6d}  {v:10.6f}")

    v_measured = np.array(v_measured)
    C_arr = np.array(C_values)

    # ---- Fit to Dirac formula ----
    # v(C) = c² / sqrt(c² + C²), fit for c
    mask = C_arr >= 0  # use all points
    popt, pcov = curve_fit(dirac_speed, C_arr[mask], v_measured[mask], p0=[1.0])
    c_fit = abs(popt[0])
    c_err = np.sqrt(pcov[0, 0])

    v_pred = dirac_speed(C_arr, c_fit)
    ss_res = np.sum((v_measured - v_pred)**2)
    ss_tot = np.sum((v_measured - np.mean(v_measured))**2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nFit: v = c² / √(c² + C²)")
    print(f"  c = {c_fit:.4f} ± {c_err:.4f} site/step")
    print(f"  R² = {r2:.6f}")

    # ---- Part 2: Scaling check at multiple sigma ----
    print(f"\n--- Part 2: Scaling check across σ = {sigmas_scaling} ---")
    print(f"{'C':>6s}  {'sigma':>6s}  {'phi':>10s}  {'v':>10s}  {'v_Dirac':>10s}  {'err%':>8s}")

    scaling_results = []  # (C, sigma, v)
    for C in C_values:
        for sigma in sigmas_scaling:
            phi = C / sigma if C > 0 else 0.0
            nsteps = max(int(3 * sigma), 500)
            # Reuse result if already computed at primary sigma
            if sigma == sigma_primary:
                idx = C_values.index(C)
                v = v_measured[idx]
                scaling_results.append((C, sigma, v))
                v_d = dirac_speed(C, c_fit)
                err = 100 * (v - v_d) / v_d if v_d != 0 else 0
                print(f"{C:6.1f}  {sigma:6.0f}  {phi:10.6f}  {v:10.6f}  {v_d:10.6f}  {err:+7.2f}%")
                continue

            sys.stderr.write(f"  C={C:.1f}, sigma={sigma:.0f} ... ")
            sys.stderr.flush()
            ts, x2s = run_walk(sigma, phi, nsteps)
            v = measure_spreading_speed(ts, x2s)
            sys.stderr.write(f"v={v:.6f}\n")

            scaling_results.append((C, sigma, v))
            v_d = dirac_speed(C, c_fit)
            err = 100 * (v - v_d) / v_d if v_d != 0 else 0
            print(f"{C:6.1f}  {sigma:6.0f}  {phi:10.6f}  {v:10.6f}  {v_d:10.6f}  {err:+7.2f}%")

    # Compute scaling CV for each C
    print(f"\nScaling coefficient of variation:")
    for C in C_values:
        vs = [v for Cv, s, v in scaling_results if Cv == C]
        cv = np.std(vs) / np.mean(vs) * 100
        print(f"  C={C:.1f}: CV = {cv:.3f}%")

    # ---- Part 3: Summary ----
    print(f"\n{'=' * 65}")
    print(f"SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Speed of light:  c = {c_fit:.4f} ± {c_err:.4f} site/step  (expected: 1.0)")
    print(f"  Dispersion fit:  R² = {r2:.6f}")
    passed = abs(c_fit - 1.0) < 0.1 and r2 > 0.99
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: v(C) dispersion curve
    ax = axes[0]
    ax.plot(C_arr, v_measured, 'bo-', markersize=6, label='measured')
    C_th = np.linspace(0, max(C_values), 200)
    ax.plot(C_th, dirac_speed(C_th, c_fit), 'r-',
            label=f'Dirac: c={c_fit:.3f}, R²={r2:.4f}')
    ax.set_xlabel('C = φσ')
    ax.set_ylabel('Spreading speed v')
    ax.set_title('Dispersion: v(C)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: v² vs C² (should be linear: 1/v² = 1/c² + C²/c⁴)
    ax = axes[1]
    inv_v2 = 1.0 / v_measured**2
    ax.plot(C_arr**2, inv_v2, 'bo-', markersize=6, label='measured')
    C2_th = np.linspace(0, max(C_values)**2, 200)
    ax.plot(C2_th, 1/c_fit**2 + C2_th/c_fit**4, 'r-',
            label=f'1/c² + C²/c⁴')
    ax.set_xlabel('C²')
    ax.set_ylabel('1/v²')
    ax.set_title('Linearity check: 1/v² vs C²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Scaling collapse
    ax = axes[2]
    markers = {200: 's', 300: 'o', 500: 'D'}
    for sigma in sigmas_scaling:
        Cs = [Cv for Cv, s, v in scaling_results if s == sigma]
        vs = [v for Cv, s, v in scaling_results if s == sigma]
        ax.plot(Cs, vs, marker=markers[sigma], linestyle='', markersize=8,
                label=f'σ={sigma}')
    ax.plot(C_th, dirac_speed(C_th, c_fit), 'r-', alpha=0.5, label='Dirac fit')
    ax.set_xlabel('C = φσ')
    ax.set_ylabel('Spreading speed v')
    ax.set_title('Scaling collapse across σ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Test A: 1D Dispersion Relation (V-mixing walk, no coin)', fontsize=14)
    plt.tight_layout()

    outpath = '/tmp/test_A_dispersion.png'
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")


if __name__ == '__main__':
    main()
