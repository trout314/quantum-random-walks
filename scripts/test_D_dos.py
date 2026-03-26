#!/usr/bin/env python3
"""
Test D: Density of States

Verifies:
  1. Massless (phi=0): DOS is approximately flat (constant)
  2. Massive (phi>0): DOS has a gap at |E| < m
  3. Van Hove singularity: DOS peaks near E = m
  4. Compare DOS shape to 1D Dirac prediction:
     rho(E) = E / sqrt(E^2 - m^2) for |E| > m (diverges at E=m)

Usage: python3 scripts/test_D_dos.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.fourier_dispersion import build_chain, build_walk_operator


def dirac_dos_1d(E, m):
    """1D massive Dirac DOS: rho(E) = |E| / sqrt(E^2 - m^2) for |E| > m."""
    result = np.zeros_like(E)
    mask = np.abs(E) > m
    result[mask] = np.abs(E[mask]) / np.sqrt(E[mask]**2 - m**2)
    return result


def main():
    N = 1000  # larger chain for better DOS resolution
    pat = [1, 3, 0, 2]

    print("=" * 65)
    print("Test D: Density of States")
    print("=" * 65)
    print(f"N = {N} sites ({4*N} eigenvalues)")

    print("\nBuilding chain...")
    positions, dirs_all, face_idx, taus = build_chain(N, pat)

    phis = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    spectra = {}

    for phi in phis:
        print(f"  phi={phi:.3f} ... ", end='', flush=True)
        W = build_walk_operator(N, taus, phi)
        eigenvalues = np.linalg.eigvals(W)
        E = np.abs(np.angle(eigenvalues))
        spectra[phi] = np.sort(E)
        gap = E.min()
        m_pred = 0.855 * phi
        print(f"gap={gap:.5f}, predicted m={m_pred:.5f}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, phi in enumerate(phis):
        ax = axes[idx // 3][idx % 3]
        E = spectra[phi]
        m = 0.855 * phi

        # Histogram DOS
        nbins = 80
        counts, bin_edges = np.histogram(E, bins=nbins, range=(0, np.pi))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]
        dos = counts / (len(E) * bin_width)  # normalize

        ax.bar(bin_centers, dos, width=bin_width * 0.9, alpha=0.6,
               color='blue', label='walk DOS')

        # Dirac prediction (normalized to match)
        if m > 0.001:
            E_th = np.linspace(m * 1.01, np.pi, 500)
            dos_th = dirac_dos_1d(E_th, m)
            # Normalize: integral of dos_th over [m, pi] should match
            # the fraction of states above m
            n_above_m = np.sum(E > m)
            integral_th = np.trapezoid(dos_th, E_th)
            if integral_th > 0:
                dos_th *= (n_above_m / len(E)) / (integral_th * bin_width)
            ax.plot(E_th, dos_th, 'r-', linewidth=2, alpha=0.7,
                    label=f'Dirac: m={m:.4f}')
            ax.axvline(m, color='r', linestyle=':', alpha=0.5)

        ax.set_xlabel('|E|')
        ax.set_ylabel('DOS')
        ax.set_title(f'$\\phi$={phi} (m={m:.4f})')
        ax.set_xlim(0, min(np.pi, max(0.5, 5 * m + 0.1)))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test D: Density of States (N={N})', fontsize=14)
    plt.tight_layout()
    out = '/tmp/test_D_dos.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    # ---- Summary statistics ----
    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'phi':>8s}  {'gap':>8s}  {'m_pred':>8s}  {'gap/m':>8s}  {'states_in_gap':>14s}")
    for phi in phis:
        E = spectra[phi]
        m = 0.855 * phi
        gap = E.min()
        n_in_gap = np.sum(E < m * 0.9) if m > 0.001 else 'n/a'
        ratio = gap / m if m > 0.001 else 'n/a'
        print(f"{phi:8.3f}  {gap:8.5f}  {m:8.5f}  {str(ratio):>8s}  {str(n_in_gap):>14s}")


if __name__ == '__main__':
    main()
