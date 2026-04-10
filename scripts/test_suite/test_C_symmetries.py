#!/usr/bin/env python3
"""
Test C: 1D Symmetries

Verifies:
  1. ±E symmetry: eigenvalues come in (E, -E) pairs
  2. Spin degeneracy: each |E| level is (at least) 2-fold degenerate
  3. Left/right symmetry: R and L spiral walks give mirror-image densities

Usage: python3 scripts/test_C_symmetries.py
"""
import subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dispersion'))
from fourier_dispersion import build_walk_operator
from src.helix_geometry import build_taus

WALK_BIN = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'dlang', 'build', 'walk_1d_d')


def run_walk_density(sigma, mix_phi, nsteps, spiral=0):
    """Run walk and read the density file."""
    cmd = [WALK_BIN, '0', str(sigma), str(nsteps),
           '3', '0', '0.0', str(mix_phi), str(spiral), '0']
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return np.loadtxt('/tmp/walk_1d_density.dat')


def main():
    print("=" * 65)
    print("Test C: 1D Symmetries")
    print("=" * 65)

    # ---- Part 1: ±E symmetry and spin degeneracy from spectrum ----
    print("\n--- Part 1: Spectral symmetries ---")

    N = 500
    pat = [1, 3, 0, 2]
    taus = build_taus(N)

    phis = [0.0, 0.04, 0.08, 0.20]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for idx, phi in enumerate(phis):
        ax = axes[idx // 2][idx % 2]

        W = build_walk_operator(N, taus, phi)
        eigenvalues = np.linalg.eigvals(W)
        E = np.angle(eigenvalues)
        E_sorted = np.sort(E)

        # Test ±E symmetry: for each E, check that -E also exists
        E_pos = np.sort(E[E >= 0])
        E_neg = np.sort(-E[E < 0])  # negate so we can compare magnitudes

        # Match positive and negative eigenvalues
        if len(E_pos) == len(E_neg):
            pm_diffs = np.abs(E_pos - E_neg)
            max_pm_diff = pm_diffs.max()
            mean_pm_diff = pm_diffs.mean()
        else:
            # Different counts — match closest
            max_pm_diff = -1
            mean_pm_diff = -1

        # Test spin degeneracy: sort |E| and check consecutive differences
        E_abs = np.sort(np.abs(E))
        deg_diffs = np.abs(np.diff(E_abs))
        # In groups of 2: every other diff should be ~0
        pair_diffs = deg_diffs[::2]  # diff between 1st and 2nd in each pair
        inter_diffs = deg_diffs[1::2]  # diff between pairs

        # Count nearly-degenerate pairs
        n_degen = np.sum(pair_diffs < 1e-8)
        n_total_pairs = len(pair_diffs)

        print(f"\n  phi={phi}:")
        print(f"    ±E symmetry: max|E+ - E-| = {max_pm_diff:.2e}, mean = {mean_pm_diff:.2e}")
        print(f"    Spin degeneracy: {n_degen}/{n_total_pairs} pairs degenerate (< 1e-8)")

        # Plot: sorted eigenvalues and degeneracy structure
        ax.plot(range(len(E_sorted)), E_sorted, 'b.', markersize=1)
        ax.axhline(0, color='k', linewidth=0.5)
        m = 0.855 * phi
        if m > 0:
            ax.axhline(m, color='r', linestyle='--', alpha=0.5, label=f'+m={m:.4f}')
            ax.axhline(-m, color='r', linestyle='--', alpha=0.5, label=f'-m')
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('E (quasienergy)')
        ax.set_title(f'$\\phi$={phi}: ±E max diff={max_pm_diff:.1e}, '
                     f'degen={n_degen}/{n_total_pairs}')
        if m > 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test C Part 1: Spectral symmetries (N={N})', fontsize=14)
    plt.tight_layout()
    out1 = '/tmp/test_C_spectrum.png'
    plt.savefig(out1, dpi=150)
    print(f"\n  Plot saved to {out1}")

    # ---- Part 2: Left/right symmetry (R vs L spiral) ----
    print("\n--- Part 2: R/L mirror symmetry ---")

    sigma = 200
    phi = 0.005  # C = 1.0
    nsteps = 1200

    print(f"  Running R and L spirals: sigma={sigma}, phi={phi}, t={nsteps}")
    dens_R = run_walk_density(sigma, phi, nsteps, spiral=0)
    # Need to copy density file before L spiral overwrites it
    import shutil
    shutil.copy('/tmp/walk_1d_density.dat', '/tmp/walk_1d_density_R.dat')
    dens_L = run_walk_density(sigma, phi, nsteps, spiral=1)

    site_R = dens_R[:, 0]
    site_L = dens_L[:, 0]
    prob_R = dens_R[:, 2]
    prob_L = dens_L[:, 2]

    # For mirror symmetry: R density at site x should equal L density at site -x
    # Build interpolation for comparison
    lo = int(max(site_R.min(), site_L.min()))
    hi = int(min(site_R.max(), site_L.max()))

    # R at site x vs L at site -x
    idx_R = (site_R >= lo) & (site_R <= hi)
    idx_L_flip = (-site_L >= lo) & (-site_L <= hi)

    r_sites = site_R[idx_R]
    r_prob = prob_R[idx_R]
    l_flip_sites = -site_L[idx_L_flip][::-1]
    l_flip_prob = prob_L[idx_L_flip][::-1]

    # Interpolate L(flipped) onto R's grid
    l_interp = np.interp(r_sites, l_flip_sites, l_flip_prob, left=0, right=0)

    # RMS difference
    mask = r_prob > 1e-12  # only compare where there's probability
    rms = np.sqrt(np.mean((r_prob[mask] - l_interp[mask])**2))
    max_diff = np.max(np.abs(r_prob[mask] - l_interp[mask]))
    mean_prob = np.mean(r_prob[mask])

    print(f"  RMS(R(x) - L(-x)) = {rms:.2e}")
    print(f"  Max|R(x) - L(-x)| = {max_diff:.2e}")
    print(f"  Mean probability = {mean_prob:.2e}")
    print(f"  Relative RMS = {rms/mean_prob:.2e}")

    # Also compute average (R + L_flipped) / 2
    avg_prob = 0.5 * (r_prob + l_interp)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    ax.plot(site_R, prob_R, 'b-', linewidth=0.8, label='R spiral')
    ax.plot(site_L, prob_L, 'r-', linewidth=0.8, label='L spiral')
    ax.plot(-site_L, prob_L, 'r--', linewidth=0.8, alpha=0.5, label='L flipped')
    ax.set_xlabel('Site')
    ax.set_ylabel('P(x)')
    ax.set_title(f'R vs L density (C={phi*sigma:.1f}, t={nsteps})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(r_sites, r_prob, 'b-', linewidth=0.8, alpha=0.5, label='R')
    ax.plot(r_sites, l_interp, 'r--', linewidth=0.8, alpha=0.5, label='L flipped')
    ax.plot(r_sites, avg_prob, 'k-', linewidth=1.5, label='Average (symmetric)')
    ax.set_xlabel('Site')
    ax.set_ylabel('P(x)')
    ax.set_title(f'Mirror symmetry check\nRMS diff = {rms:.2e}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig2.suptitle(f'Test C Part 2: R/L mirror symmetry (σ={sigma}, φ={phi}, t={nsteps})',
                  fontsize=13)
    plt.tight_layout()
    out2 = '/tmp/test_C_symmetry.png'
    plt.savefig(out2, dpi=150)
    print(f"  Plot saved to {out2}")

    # ---- Summary ----
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    pm_pass = max_pm_diff < 1e-10 if max_pm_diff >= 0 else False
    deg_pass = n_degen > n_total_pairs * 0.95
    mirror_pass = rms / mean_prob < 0.01

    print(f"  ±E symmetry:      {'PASS' if pm_pass else 'FAIL'} (max diff = {max_pm_diff:.1e})")
    print(f"  Spin degeneracy:  {'PASS' if deg_pass else 'FAIL'} ({n_degen}/{n_total_pairs} pairs)")
    print(f"  R/L mirror:       {'PASS' if mirror_pass else 'FAIL'} (rel RMS = {rms/mean_prob:.1e})")


if __name__ == '__main__':
    main()
