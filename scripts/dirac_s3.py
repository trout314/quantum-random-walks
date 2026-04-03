#!/usr/bin/env python3
"""
Massless Dirac propagator on S³ from spectral decomposition.

The Dirac operator on S³ (radius R) has eigenvalues ±(l + 3/2)/R
with degeneracy (l+1)(l+2) for each sign, l = 0, 1, 2, ...

The radial eigenfunctions (depending only on geodesic angle θ from
the source point) involve Gegenbauer polynomials. For a delta-function
source, the probability density p(θ, t) is computed from the spectral sum.

On S³, the "radial" coordinate is the geodesic angle θ ∈ [0, π].
The volume element is sin²(θ) dθ (× solid angle of S²).
"""
import numpy as np
from scipy.special import gegenbauer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dirac_eigenvalues_s3(L_max, R=1.0):
    """Return (eigenvalue, degeneracy) pairs for the Dirac operator on S³."""
    evals = []
    for l in range(L_max + 1):
        lam = (l + 1.5) / R
        deg = (l + 1) * (l + 2)
        evals.append((lam, deg))
        evals.append((-lam, deg))
    return evals


def radial_eigenfunction_s3(l, theta):
    """
    Radial part of the Dirac eigenfunction on S³ at geodesic angle θ.

    For the Dirac operator on S³, the radial functions are related to
    Gegenbauer (ultraspherical) polynomials C_l^α(cos θ) with α = 1
    for the S³ geometry (since dim S³ = 3, α = (dim-1)/2 = 1).

    The properly normalized radial function for a point source is:
    f_l(θ) = C_l^1(cos θ) × normalization

    where C_l^1 are Gegenbauer polynomials (= Chebyshev U polynomials of 2nd kind).
    """
    # Gegenbauer polynomial C_l^1(cos θ) = U_l(cos θ) = sin((l+1)θ)/sin(θ)
    # This is the Chebyshev polynomial of the second kind
    x = np.cos(theta)
    # Use scipy's gegenbauer for general case
    Cl = gegenbauer(l, 1.0)
    return Cl(x)


def dirac_propagator_s3(theta, t, R=1.0, L_max=50, mass=0.0):
    """
    Probability density at geodesic angle θ, time t, for a point source
    on S³ of radius R.

    For massless Dirac: eigenvalues ±(l + 3/2)/R
    For massive Dirac: eigenvalues ±√((l + 3/2)²/R² + m²)

    The propagator (summed over all modes) is:
    K(θ, t) = Σ_l (l+1)(l+2)/(4π²) × f_l(θ) × [e^{-iE_l t} + e^{iE_l t}]

    where f_l(θ) is the radial eigenfunction.

    The probability density is |K(θ,t)|² × sin²(θ) (volume element).
    """
    # Avoid θ = 0 and θ = π singularities
    theta = np.clip(theta, 1e-10, np.pi - 1e-10)

    K = np.zeros_like(theta, dtype=complex)

    for l in range(L_max + 1):
        E_l = np.sqrt((l + 1.5)**2 / R**2 + mass**2)
        deg = (l + 1) * (l + 2)
        f_l = radial_eigenfunction_s3(l, theta)

        # Sum over positive and negative energy states
        # Phase factor for time evolution
        K += deg * f_l * 2 * np.cos(E_l * t)

    # Normalize: total probability should be 1
    # The volume element on S³ is sin²(θ) dθ dΩ₂ where dΩ₂ = 4π (S² solid angle)
    # So ∫ |K|² sin²(θ) 4π dθ should give 1... but we need careful normalization

    return K


def compare_walk_dirac(walk_data, R, L_max=30, mass=0.0):
    """
    Compare walk probability distribution with Dirac on S³.

    walk_data: dict with 'tet_dist', 'tet_counts', 'p_by_dist' at various times
    R: effective radius of S³
    """
    max_d = walk_data['max_dist']

    # Map tet BFS distance to geodesic angle on S³
    # BFS distance d ∈ [0, max_d], geodesic angle θ ∈ [0, π]
    # Linear mapping: θ = π × d / max_d
    theta_d = np.pi * np.arange(max_d + 1) / max_d

    # Volume element: number of tets at distance d, normalized
    tet_counts = walk_data['tet_counts']

    # For Dirac on S³: volume at angle θ is sin²(θ)
    # For the triangulation: it's tet_counts[d]
    # These should be proportional if the triangulation is uniform

    print("\nDistance → angle mapping:")
    print(f"{'d':>3} {'θ/π':>6} {'n_tets':>7} {'sin²θ':>7} {'ratio':>7}")
    for d in range(max_d + 1):
        th = theta_d[d]
        s2 = np.sin(th)**2
        ratio = tet_counts[d] / max(s2, 1e-10) if tet_counts[d] > 0 else 0
        print(f"{d:3d} {th/np.pi:6.3f} {tet_counts[d]:7d} {s2:7.4f} {ratio:7.1f}")

    # Compute Dirac propagator at each timestep
    theta_fine = np.linspace(0.01, np.pi - 0.01, 500)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, t_step in enumerate(walk_data['times'][:6]):
        ax = axes[idx // 3, idx % 3]

        # Walk data: probability per tet at this time
        p_walk = walk_data['p_by_dist'][t_step]
        # Normalize by volume (tet count) to get density
        dens_walk = np.zeros(max_d + 1)
        for d in range(max_d + 1):
            if tet_counts[d] > 0:
                dens_walk[d] = p_walk[d] / tet_counts[d]

        # Dirac propagator
        K = dirac_propagator_s3(theta_fine, t_step, R=R, L_max=L_max, mass=mass)
        # Probability density (unnormalized)
        p_dirac = np.abs(K)**2

        # Normalize both to have the same total
        if np.sum(dens_walk) > 1e-15:
            dens_walk_n = dens_walk / np.max(dens_walk)
        else:
            dens_walk_n = dens_walk

        if np.max(p_dirac) > 1e-15:
            p_dirac_n = p_dirac / np.max(p_dirac)
        else:
            p_dirac_n = p_dirac

        # Plot
        ax.bar(theta_d / np.pi, dens_walk_n, width=0.03, alpha=0.6,
               color='blue', label='Walk (tet-level)')
        ax.plot(theta_fine / np.pi, p_dirac_n, 'r-', lw=2, alpha=0.7,
                label='Dirac on S³')
        ax.set_xlabel('θ/π')
        ax.set_ylabel('Density (normalized)')
        ax.set_title(f't = {t_step}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.15)

    plt.suptitle(f'Walk vs Dirac on S³ (R={R:.2f}, L_max={L_max})', fontsize=14)
    plt.tight_layout()
    out = '/tmp/walk_vs_dirac_s3.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved to {out}")
    return fig


def main():
    """Run the walk on the 600-cell and compare to Dirac on S³."""
    import ctypes, os, sys
    from collections import deque
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from triangulation import Triangulation

    tri = Triangulation.load('data/600cell.mfd')
    print(f"600-cell: {tri.n_tets} tets, {tri.n_verts} verts")

    lib = ctypes.CDLL('dlang/build/quantum_walk.so')
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_dp = ctypes.POINTER(ctypes.c_double)
    lib.manifold_load_triangulation.argtypes = [c_ip, c_ip, ctypes.c_int]
    lib.manifold_build_lattice.restype = ctypes.c_int
    lib.manifold_nsites.restype = ctypes.c_int
    lib.manifold_step.argtypes = [ctypes.c_double]
    lib.manifold_step.restype = ctypes.c_double
    lib.manifold_set_psi_bulk.argtypes = [c_dp, c_dp]
    lib.manifold_get_psi.argtypes = [c_dp, c_dp]
    lib.manifold_get_site_tets.argtypes = [c_ip]
    lib.manifold_site_probs.argtypes = [c_dp]
    lib.manifold_site_probs.restype = ctypes.c_double

    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    nbrs_flat = np.array(tri.neighbors, dtype=np.int32).flatten()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(c_ip), nbrs_flat.ctypes.data_as(c_ip), tri.n_tets)
    nsites = lib.manifold_build_lattice()
    dim = 4 * nsites

    tets_arr = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(tets_arr.ctypes.data_as(c_ip))
    tet_sites = {}
    for s in range(nsites):
        t = tets_arr[s]
        if t not in tet_sites: tet_sites[t] = []
        tet_sites[t].append(s)

    # BFS distances
    origin_tet = 0
    tet_dist = np.full(tri.n_tets, -1, dtype=np.int32)
    tet_dist[origin_tet] = 0
    q = deque([origin_tet])
    while q:
        t = q.popleft()
        for nb in tri.neighbors[t]:
            if tet_dist[nb] < 0:
                tet_dist[nb] = tet_dist[t] + 1
                q.append(nb)
    max_dist = tet_dist.max()

    tet_counts = np.array([np.sum(tet_dist == d) for d in range(max_dist + 1)])

    # Single-site IC
    mix_phi = 0.05
    chi = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    origin_site = tet_sites[origin_tet][0]

    psi_re = np.zeros(dim, dtype=np.float64)
    psi_im = np.zeros(dim, dtype=np.float64)
    psi_re[4*origin_site:4*origin_site+4] = chi.real
    psi_im[4*origin_site:4*origin_site+4] = chi.imag
    lib.manifold_set_psi_bulk(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))

    # Evolve and record
    site_probs = np.zeros(nsites, dtype=np.float64)
    times = list(range(13))
    p_by_dist = {}

    for t in range(max(times) + 1):
        lib.manifold_site_probs(site_probs.ctypes.data_as(c_dp))
        if t in times:
            p = np.zeros(max_dist + 1)
            for tid in range(tri.n_tets):
                d = tet_dist[tid]
                for s in tet_sites[tid]:
                    p[d] += site_probs[s]
            p_by_dist[t] = p
        if t < max(times):
            lib.manifold_step(mix_phi)

    walk_data = {
        'max_dist': max_dist,
        'tet_counts': tet_counts,
        'p_by_dist': p_by_dist,
        'times': times,
    }

    # Effective radius: map BFS diameter 15 to geodesic distance π
    # Each walk step moves ~2 tet-hops (we saw this), and the walk speed
    # should be ~1 in units of c. So the effective radius satisfies:
    # v_walk (tet-hops/step) × Δθ_per_tet-hop = c = 1/R (natural units)
    # With v_walk ≈ 0.33 and Δθ = π/15 per tet-hop:
    # Effective time unit: 1 walk step corresponds to Δt = Δθ/v_dirac
    # Let's try R = max_dist / π (so θ = π at d = max_dist)
    # and adjust the Dirac time to match: t_dirac = t_walk × v_tet × (π/max_dist)
    #
    # Actually, let's just try a few values of R and see which matches

    print("\nComparing walk to Dirac on S³...")
    print(f"Tet BFS diameter: {max_dist}")
    print(f"Tet counts: {tet_counts}")

    # The effective radius in walk units: if d_BFS = 15 corresponds to θ = π,
    # and the Dirac propagation speed is 1, then R_eff = max_dist / π ≈ 4.77
    # But the walk moves ~2 tet-hops per step, so t_Dirac ≈ t_walk × 2 / R_eff
    # Let's let R be a free parameter and see what fits

    for R_trial in [2.0, 3.0, 4.0, 5.0]:
        print(f"\n--- R = {R_trial} ---")
        compare_walk_dirac(walk_data, R=R_trial, L_max=40, mass=0.0)


if __name__ == '__main__':
    main()
