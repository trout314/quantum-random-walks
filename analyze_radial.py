#!/usr/bin/env python3
"""
Compare radial probability density from quantum walk vs free Dirac equation.

The free Dirac equation in 3D: i∂ψ/∂t = (-i c α·∇ + β m c²) ψ

For a Gaussian initial spinor ψ(r,0) = (2πσ²)^{-3/4} exp(-r²/4σ²) χ,
the radial probability density at time t can be computed exactly in
momentum space:
  ψ(p,t) = ψ̃(p) exp(-iEt),  E = sqrt(c²p² + m²c⁴)

We need to determine the effective c and m from the walk parameters.
For the tetrahedral walk with coin angle θ:
  - Effective speed: c_eff = (2/3) * cos(θ)  [step_length * cos(θ)]
  - Effective mass parameter: m_eff ~ sin(θ) / step_length

The radial probability density is:
  ρ(r,t) = ∫ |ψ(r,t)|² dΩ = 4π r² |ψ(r,t)|²
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def load_radial_hist(filename):
    """Load radial histogram data from walk_evolve output."""
    data = defaultdict(list)  # t -> list of (r, density, prob, nsites)
    current_t = None
    current_norm = None
    norms = {}
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                parts = line.split()
                for p in parts:
                    if p.startswith('t='):
                        current_t = int(p.split('=')[1])
                    if p.startswith('norm='):
                        current_norm = float(p.split('=')[1])
                        norms[current_t] = current_norm
                continue
            parts = line.split()
            if len(parts) >= 3:
                t = int(parts[0])
                r = float(parts[1])
                prob = float(parts[2])
                # Old format had 5 cols (t, r, density, prob, nsites)
                # New format has 3 cols (t, r, prob)
                if len(parts) >= 5:
                    prob = float(parts[3])  # old format: prob is 4th column
                data[t].append((r, prob))
    return data, norms


def dirac_radial_density(r_arr, t, sigma, theta, c_eff=None, m_eff=None):
    """
    Compute radial probability density for free Dirac equation.

    The initial state is a Gaussian wavepacket with spinor (1,0,0,0):
      ψ(r,0) ∝ exp(-r²/4σ²) [1,0,0,0]^T

    In momentum space, this is:
      ψ̃(p) ∝ exp(-σ²p²) [1,0,0,0]^T

    Time evolution: each momentum component evolves with exp(-iEt/ℏ).
    For the positive-energy spinor with spin up, the Dirac propagator gives:
      ψ(r,t) = (1/2π²) ∫₀^∞ p² dp sin(pr)/(pr) exp(-σ²p²)
               × [cos(Et) - i sin(Et) (mc²/E), 0, -i sin(Et)(cp_z/E), ...]

    For the radial probability density (averaged over angles), only the
    scalar parts contribute, and we get:
      ρ(r,t) = 4πr² |ψ(r,t)|²

    We compute this numerically via Fourier-Bessel transform.
    """
    if c_eff is None:
        step_len = 2.0/3.0
        c_eff = step_len * np.cos(theta)
    if m_eff is None:
        step_len = 2.0/3.0
        m_eff = np.sin(theta) / step_len

    # Momentum grid
    dp = 0.005
    p_max = 8.0 / sigma  # enough to capture the Gaussian
    p = np.arange(dp/2, p_max, dp)

    # Energy
    E = np.sqrt(c_eff**2 * p**2 + m_eff**2 * c_eff**4)

    # Gaussian in momentum space (unnormalized)
    # Position space: exp(-r²/2σ²), Fourier transform: exp(-σ²p²/2)
    gauss_p = np.exp(-sigma**2 * p**2 / 2)

    # Time evolution phases
    cos_Et = np.cos(E * t)
    sin_Et = np.sin(E * t)

    # For a spin-up positive-energy state, the probability involves
    # |cos(Et) + i(mc²/E)sin(Et)|² + |cp/E sin(Et)|² summed over spinor components
    # The upper components get factor: cos²(Et) + (mc²/E)² sin²(Et)
    # The lower components get factor: (cp/E)² sin²(Et)
    # Total: cos²(Et) + sin²(Et) [(mc²/E)² + (cp/E)²] = cos²(Et) + sin²(Et) = 1
    # So norm is preserved (as expected for unitary evolution).
    # But the RADIAL distribution changes because different p-components
    # oscillate at different rates.

    # The wavefunction in position space (radial part, for isotropic initial state):
    # ψ(r) = 1/(2π²r) ∫ p sin(pr) ψ̃(p) exp(-iEt) dp
    #
    # For the 4-spinor, upper components:
    #   f_upper(r,t) = 1/(2π²r) ∫ p sin(pr) gauss(p) [cos(Et) - i(mc²/E)sin(Et)] dp
    # Lower components (the σ·p̂ part averages to give radial term):
    #   f_lower(r,t) = 1/(2π²r) ∫ p sin(pr) gauss(p) [-i(cp/E)sin(Et)] dp
    #   (but this depends on angle; for radial density we need |f|² averaged over solid angle)
    #
    # For the upper two spinor components (spin up+down combined for isotropic case):
    # After angular averaging, the radial density from the upper components:
    #   ρ_upper(r,t) = |1/(2π²r) ∫ p sin(pr) gauss(p) cos_phase(p,t) dp|²
    # where cos_phase = cos(Et) - i(mc²/E)sin(Et)  (has |cos_phase|² = cos²+m²c⁴/E² sin²)
    #
    # And from the lower components:
    #   ρ_lower(r,t) = (1/3) |1/(2π²r) ∫ p² sin(pr)/(pr) gauss(p) (cp/E)sin(Et) dp|² × something

    # Actually, let me just do this properly with the full spinor structure.
    # For initial state χ = (1,0,0,0), the positive-energy projector gives:
    # The time-evolved state in position space involves a Fourier-Bessel integral.

    # Simplification for isotropic initial state:
    # The radial probability density (integrating over solid angle) is:
    # ρ(r,t) = (2/π) r² × { |I_0(r,t)|² + |I_1(r,t)|² }
    # where
    #   I_0(r,t) = ∫ p² j_0(pr) gauss(p) [cos(Et) + i(mc⁴/E²)sin(Et) ...] dp
    # This is getting complicated. Let me just numerically compute the two integrals.

    # Use the standard decomposition for Dirac propagator.
    # For initial spinor (1,0,0,0) at rest, positive energy:

    rho = np.zeros_like(r_arr)

    for ir, r in enumerate(r_arr):
        if r < 1e-10:
            # At origin: use j_0(0)=1, j_1(0)=0
            # Upper component integral
            integrand_upper = p**2 * gauss_p * dp
            I_cos = np.sum(integrand_upper * cos_Et)
            I_msin = np.sum(integrand_upper * (m_eff * c_eff**2 / E) * sin_Et)
            rho_upper = (I_cos**2 + I_msin**2) / (2*np.pi**2)**2
            rho[ir] = 4 * np.pi * 1e-10 * rho_upper  # ~ 0
            continue

        pr = p * r
        j0 = np.sin(pr) / pr  # spherical Bessel j_0
        j1 = np.sin(pr)/pr**2 - np.cos(pr)/pr  # spherical Bessel j_1

        # Upper components: proportional to j_0
        # Real part: cos(Et), gives spreading of probability
        # The (1,0) upper spinor component:
        integrand_upper = p**2 * gauss_p * j0 * dp
        I_re = np.sum(integrand_upper * cos_Et)
        I_im = np.sum(integrand_upper * (-m_eff*c_eff**2/E) * sin_Et)
        upper_sq = (I_re**2 + I_im**2) / (2*np.pi**2)**2

        # Lower components: proportional to j_1 × (cp/E)
        integrand_lower = p**2 * gauss_p * j1 * (c_eff * p / E) * dp
        J_im = np.sum(integrand_lower * sin_Et)
        lower_sq = J_im**2 / (2*np.pi**2)**2

        # Total radial probability density (factor 4πr² from solid angle)
        rho[ir] = 4 * np.pi * r**2 * (upper_sq + lower_sq)

    # Normalize
    dr_arr = r_arr[1] - r_arr[0] if len(r_arr) > 1 else 1.0
    norm = np.sum(rho) * dr_arr
    if norm > 0:
        rho /= norm

    return rho


def main():
    hist_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/radial.dat'
    theta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

    print(f"Loading {hist_file}, theta={theta}, sigma={sigma}")
    data, norms = load_radial_hist(hist_file)

    timesteps = sorted(data.keys())
    print(f"Timesteps: {len(timesteps)}, range [{timesteps[0]}, {timesteps[-1]}]")

    # Select a few timesteps to plot
    plot_times = [0, 5, 10, 15, 20, 30, 40, 50]
    plot_times = [t for t in plot_times if t in data]

    # Scan c_eff to find best fit at an intermediate timestep
    scan_t = plot_times[min(3, len(plot_times)-1)]  # pick t=15 or similar
    scan_rows = data[scan_t]
    scan_r = np.array([row[0] for row in scan_rows])
    scan_prob = np.array([row[1] for row in scan_rows])
    scan_dr = scan_r[1] - scan_r[0] if len(scan_r) > 1 else 1.0
    scan_rd = scan_prob / scan_dr
    scan_rd /= np.sum(scan_rd) * scan_dr  # normalize

    best_c, best_m, best_err = 0.4, 0.5, 1e10
    step_len = 2.0/3.0
    for c_try in np.arange(0.05, 1.0, 0.02):
        for m_try in np.arange(0.0, 2.0, 0.05):
            r_d = np.linspace(0.01, scan_r[-1]*0.8, 300)
            rho_d = dirac_radial_density(r_d, scan_t, sigma, theta, c_eff=c_try, m_eff=m_try)
            # Interpolate walk onto dirac grid
            walk_interp = np.interp(r_d, scan_r, scan_rd)
            err = np.sum((walk_interp - rho_d)**2)
            if err < best_err:
                best_err = err
                best_c = c_try
                best_m = m_try
    print(f"Best fit at t={scan_t}: c_eff={best_c:.3f}, m_eff={best_m:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, t in enumerate(plot_times):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Walk data
        rows = data[t]
        r_walk = np.array([row[0] for row in rows])
        prob_walk = np.array([row[1] for row in rows])

        # Normalize: convert prob_walk to probability density
        dr = r_walk[1] - r_walk[0] if len(r_walk) > 1 else 1.0
        # prob_walk is already normalized (sums to 1)
        # density = prob / (4π r² dr) is the volumetric density
        # radial prob density = prob / dr = 4π r² × volumetric density
        radial_density_walk = prob_walk / dr

        # Dirac prediction with fitted parameters
        r_dirac = np.linspace(0.01, r_walk[-1] * 0.8, 500)
        rho_dirac = dirac_radial_density(r_dirac, t, sigma, theta, c_eff=best_c, m_eff=best_m)

        # Scale Dirac to match walk norm (boundary absorption)
        walk_norm = norms.get(t, 1.0)

        # Renormalize both to unit area for shape comparison
        walk_area = np.sum(radial_density_walk) * dr
        dirac_area = np.sum(rho_dirac) * (r_dirac[1]-r_dirac[0]) if len(r_dirac) > 1 else 1.0

        rw_norm = radial_density_walk / walk_area if walk_area > 0 else radial_density_walk
        rd_norm = rho_dirac / dirac_area if dirac_area > 0 else rho_dirac

        # Plot
        ax.plot(r_walk, rw_norm, 'b-', alpha=0.7, label='Walk', linewidth=1)
        ax.plot(r_dirac, rd_norm, 'r--', alpha=0.7, label='Dirac', linewidth=1.5)
        ax.set_title(f't={t}, norm={walk_norm:.4f}')
        ax.set_xlabel('r')
        ax.set_ylabel('P(r) [normalized]')
        ax.legend(fontsize=8)
        # Set x limit to capture the action
        r95_walk = 0
        cum = 0
        for i in range(len(radial_density_walk)):
            cum += radial_density_walk[i] * dr
            if cum / walk_area > 0.99:
                r95_walk = r_walk[i]
                break
        ax.set_xlim(0, max(r95_walk * 1.5, 10))

    plt.suptitle(f'Radial probability density: Walk vs Dirac (θ={theta}, σ={sigma})', fontsize=14)
    plt.tight_layout()
    plt.savefig('/tmp/radial_comparison.png', dpi=150)
    print("Saved /tmp/radial_comparison.png")

    # Also plot r² vs t comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Load summary data (from stdout, piped to file or from stdin of hist file)
    summary_file = hist_file.replace('radial', 'evolve').replace('.dat', '.txt')
    import os
    if not os.path.exists(summary_file):
        # Try to extract from the same run's stdout
        summary_file = '/tmp/evolve_out.txt'
    ts, norms_arr, r2s = [], [], []
    # Read the walk output directly — it was printed to stdout
    # The radial hist file only has histogram data. We need the summary.
    # For now, reconstruct from the histogram norms.
    for t in sorted(data.keys()):
        ts.append(t)
        norms_arr.append(norms.get(t, 1.0))
        # Compute r2 from histogram
        rows_t = data[t]
        r_arr = np.array([row[0] for row in rows_t])
        p_arr = np.array([row[1] for row in rows_t])
        total_p = np.sum(p_arr)
        if total_p > 0:
            r2s.append(np.sum(p_arr * r_arr**2) / total_p)
        else:
            r2s.append(0)

    ts = np.array(ts)
    norms_arr = np.array(norms_arr)
    r2s = np.array(r2s)

    # Fit effective velocity from conditional <r²>
    # Use early-to-mid times where the wavepacket is still well-contained
    fit_max = min(15, len(ts)-1)
    t_fit = ts[:fit_max+1]
    r2_fit = r2s[:fit_max+1]
    # Fit <r²> = a + b*t + c*t² (allow linear + quadratic)
    coeffs = np.polyfit(t_fit, r2_fit, 2)
    v_fitted = np.sqrt(max(coeffs[0], 0))  # quadratic coefficient = v²
    r2_model = np.polyval(coeffs, ts)

    ax1.plot(ts, r2s, 'b-', label='Walk <r²> (conditional)')
    ax1.plot(ts, r2_model, 'r--', label=f'Fit: v_eff={v_fitted:.3f}')
    ax1.set_xlabel('t')
    ax1.set_ylabel('<r²> | survived')
    ax1.legend()
    ax1.set_title('Conditional <r²> vs time')

    ax2.semilogy(ts, norms_arr, 'b-')
    ax2.set_xlabel('t')
    ax2.set_ylabel('norm')
    ax2.set_title('Norm decay (boundary absorption)')

    print(f"Fitted v_eff = {v_fitted:.4f} (from conditional <r²>)")
    print(f"For reference: step_len = {2/3:.4f}, step_len*cos(theta) = {2/3*np.cos(theta):.4f}")

    plt.tight_layout()
    plt.savefig('/tmp/r2_comparison.png', dpi=150)
    print("Saved /tmp/r2_comparison.png")


if __name__ == '__main__':
    main()
