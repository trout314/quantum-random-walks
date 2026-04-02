#!/usr/bin/env python3
"""
Compare tet-level spreading under different initial conditions.

IC A: Site-centered — wavepacket from manifold_init_wavepacket (original)
IC B: Tet-symmetric — equal amplitude at all 12 sites of origin tet,
      Gaussian envelope using tet-BFS distance
IC C: Single-site — amplitude at exactly 1 site, no Gaussian

All evolved on the same manifold, compared at tet level.
"""
import ctypes, numpy as np, os, sys, time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from triangulation import Triangulation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_lib():
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'dlang', 'build', 'quantum_walk.so')
    lib = ctypes.CDLL(lib_path)
    c_dp = ctypes.POINTER(ctypes.c_double)
    c_ip = ctypes.POINTER(ctypes.c_int)
    lib.manifold_load_triangulation.argtypes = [c_ip, c_ip, ctypes.c_int]
    lib.manifold_build_lattice.restype = ctypes.c_int
    lib.manifold_nsites.restype = ctypes.c_int
    lib.manifold_nchains.restype = ctypes.c_int
    lib.manifold_norm2.restype = ctypes.c_double
    lib.manifold_step.argtypes = [ctypes.c_double]
    lib.manifold_step.restype = ctypes.c_double
    lib.manifold_zero_psi.restype = None
    lib.manifold_set_psi_bulk.argtypes = [c_dp, c_dp]
    lib.manifold_get_psi.argtypes = [c_dp, c_dp]
    lib.manifold_get_site_tets.argtypes = [c_ip]
    lib.manifold_init_wavepacket.argtypes = [
        ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.manifold_site_probs.argtypes = [c_dp]
    lib.manifold_site_probs.restype = ctypes.c_double
    lib.manifold_has_chain.argtypes = [ctypes.c_int, ctypes.c_bool]
    lib.manifold_has_chain.restype = ctypes.c_bool
    lib.manifold_chain_next.argtypes = [ctypes.c_int, ctypes.c_bool]
    lib.manifold_chain_next.restype = ctypes.c_int
    lib.manifold_chain_prev.argtypes = [ctypes.c_int, ctypes.c_bool]
    lib.manifold_chain_prev.restype = ctypes.c_int
    return lib


def bfs_tet_distances(tri, origin_tet):
    dist = np.full(tri.n_tets, -1, dtype=np.int32)
    dist[origin_tet] = 0
    queue = deque([origin_tet])
    while queue:
        t = queue.popleft()
        for nb in tri.neighbors[t]:
            if dist[nb] < 0:
                dist[nb] = dist[t] + 1
                queue.append(nb)
    return dist


def bfs_site_distances(lib, nsites, origin_site):
    dist = np.full(nsites, -1, dtype=np.int32)
    dist[origin_site] = 0
    queue = deque([origin_site])
    while queue:
        s = queue.popleft()
        for isR in [True, False]:
            if not lib.manifold_has_chain(s, isR):
                continue
            for nb in [lib.manifold_chain_next(s, isR), lib.manifold_chain_prev(s, isR)]:
                if nb >= 0 and dist[nb] < 0:
                    dist[nb] = dist[s] + 1
                    queue.append(nb)
    return dist


def evolve_and_measure(lib, nsites, n_tets, tet_sites, tets_arr,
                       tet_dist, n_steps, mix_phi):
    """Evolve current state and return tet-level Var(d) at each step."""
    c_dp = ctypes.POINTER(ctypes.c_double)
    site_probs = np.zeros(nsites, dtype=np.float64)
    tet_var = np.zeros(n_steps + 1)
    tet_mean = np.zeros(n_steps + 1)
    purity = np.zeros(n_steps + 1)

    for t in range(n_steps + 1):
        lib.manifold_site_probs(site_probs.ctypes.data_as(c_dp))

        # Tet probabilities
        tet_probs = np.zeros(n_tets)
        for tet_id in range(n_tets):
            for s in tet_sites[tet_id]:
                tet_probs[tet_id] += site_probs[s]

        d1 = np.sum(tet_probs * tet_dist)
        d2 = np.sum(tet_probs * tet_dist**2)
        tet_mean[t] = d1
        tet_var[t] = d2 - d1**2

        # Purity at densest tet
        peak = np.argmax(tet_probs)
        dim = 4 * nsites
        psi_re = np.zeros(dim, dtype=np.float64)
        psi_im = np.zeros(dim, dtype=np.float64)
        lib.manifold_get_psi(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
        psi = psi_re + 1j * psi_im
        rho = np.zeros((4, 4), dtype=complex)
        for s in tet_sites[peak]:
            sp = psi[4*s:4*s+4]
            rho += np.outer(sp, sp.conj())
        tr = np.real(np.trace(rho))
        purity[t] = np.real(np.trace(rho @ rho)) / max(tr**2, 1e-30) if tr > 1e-15 else 0

        if t < n_steps:
            lib.manifold_step(mix_phi)

    return tet_var, tet_mean, purity


def main():
    tri_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else
        '~/Desktop/Discrete-Differential-Geometry/standard_triangulations/equilibrated_1000.mfd')
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    tri = Triangulation.load(tri_path)
    n_tets = tri.n_tets
    print(f"Manifold: {n_tets} tets")

    lib = load_lib()
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_dp = ctypes.POINTER(ctypes.c_double)

    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    nbrs_flat = np.array(tri.neighbors, dtype=np.int32).flatten()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(c_ip), nbrs_flat.ctypes.data_as(c_ip), n_tets)
    nsites = lib.manifold_build_lattice()
    dim = 4 * nsites
    print(f"Lattice: {nsites} sites")

    # Mappings
    tets_arr = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(tets_arr.ctypes.data_as(c_ip))
    tet_sites = {}
    for s in range(nsites):
        t = tets_arr[s]
        if t not in tet_sites:
            tet_sites[t] = []
        tet_sites[t].append(s)

    origin_site = 0
    origin_tet = tets_arr[origin_site]
    tet_dist = bfs_tet_distances(tri, origin_tet).astype(float)
    site_dist = bfs_site_distances(lib, nsites, origin_site)
    print(f"Origin tet: {origin_tet}, BFS diameter: {int(tet_dist.max())}")

    chi = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

    results = {}

    # ================================================================
    # IC A: Site-centered wavepacket (original)
    # ================================================================
    print("\n--- IC A: Site-centered wavepacket (σ=5 site-BFS) ---")
    sigma_site = 5.0
    lib.manifold_init_wavepacket(origin_site, 1e10, 0.0, 0.0, 0.0)
    psi_re = np.zeros(dim, dtype=np.float64)
    psi_im = np.zeros(dim, dtype=np.float64)
    lib.manifold_get_psi(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    for sid in range(nsites):
        w = np.exp(-site_dist[sid]**2 / (2 * sigma_site**2))
        psi_re[4*sid:4*sid+4] *= w
        psi_im[4*sid:4*sid+4] *= w
    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)
    lib.manifold_set_psi_bulk(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    # Count how many sites at origin tet have significant amplitude
    psi0 = psi_re + 1j * psi_im
    dens_at_origin = [np.sum(np.abs(psi0[4*s:4*s+4])**2) for s in tet_sites[origin_tet]]
    print(f"  Sites with >1% of max at origin tet: "
          f"{sum(d > 0.01*max(dens_at_origin) for d in dens_at_origin)}/12")
    t0 = time.perf_counter()
    var_a, mean_a, pur_a = evolve_and_measure(
        lib, nsites, n_tets, tet_sites, tets_arr, tet_dist, n_steps, mix_phi)
    print(f"  {time.perf_counter()-t0:.1f}s")
    results['A: site-centered'] = (var_a, mean_a, pur_a)

    # ================================================================
    # IC B: Tet-symmetric — same spinor at all 12 sites, Gaussian in tet distance
    # ================================================================
    print("\n--- IC B: Tet-symmetric (same χ at all 12 sites, Gaussian in tet-BFS) ---")
    sigma_tet = 3.0
    psi_re[:] = 0; psi_im[:] = 0
    for tet_id in range(n_tets):
        w = np.exp(-tet_dist[tet_id]**2 / (2 * sigma_tet**2))
        for s in tet_sites[tet_id]:
            psi_re[4*s:4*s+4] = w * chi.real
            psi_im[4*s:4*s+4] = w * chi.imag
    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)
    lib.manifold_set_psi_bulk(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    psi0 = psi_re + 1j * psi_im
    dens_at_origin = [np.sum(np.abs(psi0[4*s:4*s+4])**2) for s in tet_sites[origin_tet]]
    print(f"  Sites with >1% of max at origin tet: "
          f"{sum(d > 0.01*max(dens_at_origin) for d in dens_at_origin)}/12")
    t0 = time.perf_counter()
    var_b, mean_b, pur_b = evolve_and_measure(
        lib, nsites, n_tets, tet_sites, tets_arr, tet_dist, n_steps, mix_phi)
    print(f"  {time.perf_counter()-t0:.1f}s")
    results['B: tet-symmetric'] = (var_b, mean_b, pur_b)

    # ================================================================
    # IC C: Single site, no Gaussian
    # ================================================================
    print("\n--- IC C: Single site only ---")
    psi_re[:] = 0; psi_im[:] = 0
    psi_re[4*origin_site:4*origin_site+4] = chi.real
    psi_im[4*origin_site:4*origin_site+4] = chi.imag
    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)
    lib.manifold_set_psi_bulk(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    t0 = time.perf_counter()
    var_c, mean_c, pur_c = evolve_and_measure(
        lib, nsites, n_tets, tet_sites, tets_arr, tet_dist, n_steps, mix_phi)
    print(f"  {time.perf_counter()-t0:.1f}s")
    results['C: single-site'] = (var_c, mean_c, pur_c)

    # ================================================================
    # IC D: All 12 sites at origin tet, random phases (maximally incoherent)
    # ================================================================
    print("\n--- IC D: All 12 sites, random phases (incoherent) ---")
    rng = np.random.RandomState(42)
    psi_re[:] = 0; psi_im[:] = 0
    for s in tet_sites[origin_tet]:
        phase = rng.uniform(0, 2*np.pi)
        sp = np.exp(1j * phase) * chi
        psi_re[4*s:4*s+4] = sp.real
        psi_im[4*s:4*s+4] = sp.imag
    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)
    lib.manifold_set_psi_bulk(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    t0 = time.perf_counter()
    var_d, mean_d, pur_d = evolve_and_measure(
        lib, nsites, n_tets, tet_sites, tets_arr, tet_dist, n_steps, mix_phi)
    print(f"  {time.perf_counter()-t0:.1f}s")
    results['D: random-phase'] = (var_d, mean_d, pur_d)

    # ================================================================
    # Analysis: early-time fitting
    # ================================================================
    print("\n" + "=" * 60)
    print("EARLY-TIME SPREADING ANALYSIS (t=3 to t=50)")
    print("=" * 60)

    t_arr = np.arange(n_steps + 1).astype(float)
    fit_lo, fit_hi = 3, min(50, n_steps)
    mask = (t_arr >= fit_lo) & (t_arr <= fit_hi)
    t_fit = t_arr[mask]

    for label, (var, mean, pur) in results.items():
        v_fit = var[mask]
        # Linear: Var = a + b*t
        c1 = np.polyfit(t_fit, v_fit, 1)
        r1 = np.std(v_fit - np.polyval(c1, t_fit))
        # Quadratic: Var = a + b*t + c*t²
        c2 = np.polyfit(t_fit, v_fit, 2)
        r2 = np.std(v_fit - np.polyval(c2, t_fit))

        print(f"\n  {label}:")
        print(f"    Linear:    Var = {c1[1]:.3f} + {c1[0]:.4f}·t  (resid={r1:.4f})")
        print(f"    Quadratic: Var = {c2[2]:.3f} + {c2[1]:.4f}·t + {c2[0]:.6f}·t²  (resid={r2:.4f})")
        if c2[0] > 0:
            print(f"    → BALLISTIC component: v = √c = {np.sqrt(c2[0]):.4f}")
        else:
            print(f"    → Quadratic term negative (decelerating)")
        # ΔVar/Δt at early times
        dv = np.diff(var[:21])
        print(f"    ΔVar/Δt (t=1..5):  {dv[:5]}")
        print(f"    ΔVar/Δt (t=6..10): {dv[5:10]}")
        print(f"    ΔVar/Δt (t=11..20):{dv[10:20]}")

    # ================================================================
    # Plot
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Var(d) vs time — all ICs, early time only
    ax = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for (label, (var, mean, pur)), col in zip(results.items(), colors):
        ax.plot(t_arr[:fit_hi+1], var[:fit_hi+1], '-', color=col, lw=1.5, label=label)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Var(d_tet)')
    ax.set_title(f'(a) Tet-level variance (t≤{fit_hi})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # (b) Var(d) — full time range
    ax = axes[0, 1]
    for (label, (var, mean, pur)), col in zip(results.items(), colors):
        ax.plot(t_arr, var, '-', color=col, lw=1.5, label=label)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Var(d_tet)')
    ax.set_title('(b) Tet-level variance (full)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # (c) ΔVar/Δt vs time (smoothed)
    ax = axes[1, 0]
    window = 5
    for (label, (var, mean, pur)), col in zip(results.items(), colors):
        dv = np.diff(var)
        if len(dv) > window:
            dv_smooth = np.convolve(dv, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window//2, window//2+len(dv_smooth)),
                    dv_smooth, '-', color=col, lw=1.2, label=label)
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('Time step')
    ax.set_ylabel('ΔVar/Δt (smoothed)')
    ax.set_title(f'(c) Spreading rate (window={window})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # (d) Purity
    ax = axes[1, 1]
    for (label, (var, mean, pur)), col in zip(results.items(), colors):
        ax.plot(t_arr, pur, '-', color=col, lw=1, alpha=0.7, label=label)
    ax.axhline(0.25, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Purity')
    ax.set_title('(d) Purity at densest tet')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f'IC comparison: {n_tets} tets, φ={mix_phi}', fontsize=14, y=1.01)
    plt.tight_layout()
    out = '/tmp/tet_spreading_ic.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
