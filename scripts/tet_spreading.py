#!/usr/bin/env python3
"""
Tet-level wavepacket spreading on the manifold.

Compares two views of the same walk evolution:
  (a) Site-level: p(site) = |ψ_site|², distance = BFS on site graph
  (b) Tet-level:  p(tet)  = Σ_{sites at tet} |ψ_site|², distance = BFS on dual graph

Also tracks the 4×4 density matrix at the densest tet to see
if spin coherence survives despite purity loss.
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
    lib.manifold_get_site_positions.argtypes = [c_dp]
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
    """BFS on the dual graph (tet adjacency via shared faces)."""
    n_tets = tri.n_tets
    dist = np.full(n_tets, -1, dtype=np.int32)
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
    """BFS on site adjacency (chain links)."""
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


def tet_reduced_density(psi, sites):
    """4×4 reduced density matrix from sites."""
    rho = np.zeros((4, 4), dtype=complex)
    for s in sites:
        sp = psi[4*s:4*s+4]
        rho += np.outer(sp, sp.conj())
    return rho


def main():
    tri_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else
        '~/Desktop/Discrete-Differential-Geometry/standard_triangulations/equilibrated_200.mfd')
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

    tri = Triangulation.load(tri_path)
    n_tets = tri.n_tets
    print(f"Manifold: {n_tets} tets, {tri.n_verts} verts")

    lib = load_lib()
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_dp = ctypes.POINTER(ctypes.c_double)

    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    nbrs_flat = np.array(tri.neighbors, dtype=np.int32).flatten()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(c_ip), nbrs_flat.ctypes.data_as(c_ip), n_tets)
    nsites = lib.manifold_build_lattice()
    dim = 4 * nsites
    print(f"Lattice: {nsites} sites, {lib.manifold_nchains()} chains")

    # Site → tet mapping
    tets_arr = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(tets_arr.ctypes.data_as(c_ip))
    tet_sites = {}
    for s in range(nsites):
        t = tets_arr[s]
        if t not in tet_sites:
            tet_sites[t] = []
        tet_sites[t].append(s)

    # Distances
    origin_site = 0
    origin_tet = tets_arr[origin_site]
    print(f"Origin: site {origin_site}, tet {origin_tet}")

    tet_dist = bfs_tet_distances(tri, origin_tet)
    site_dist = bfs_site_distances(lib, nsites, origin_site)
    max_tet_dist = tet_dist.max()
    print(f"Tet BFS diameter from origin: {max_tet_dist}")
    print(f"Site BFS diameter from origin: {site_dist.max()}")

    # Initialize wavepacket
    lib.manifold_init_wavepacket(origin_site, sigma, 0.0, 0.0, 0.0)

    # Modulate by site-distance Gaussian
    psi_re = np.zeros(dim, dtype=np.float64)
    psi_im = np.zeros(dim, dtype=np.float64)
    lib.manifold_get_psi(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    for sid in range(nsites):
        d = site_dist[sid]
        w = np.exp(-d**2 / (2 * sigma**2))
        psi_re[4*sid:4*sid+4] *= w
        psi_im[4*sid:4*sid+4] *= w
    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)
    lib.manifold_set_psi_bulk(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    print(f"Initial norm: {np.sqrt(lib.manifold_norm2()):.12f}")

    # Precompute tet-distance array for each site
    site_tet_dist = np.array([tet_dist[tets_arr[s]] for s in range(nsites)], dtype=np.float64)

    # Storage
    site_probs = np.zeros(nsites, dtype=np.float64)
    record_times = list(range(min(20, n_steps+1))) + list(range(20, n_steps+1, 10))
    record_times = sorted(set(t for t in record_times if t <= n_steps))

    # Tet-level observables
    tet_var = np.zeros(n_steps + 1)
    tet_mean = np.zeros(n_steps + 1)
    site_var = np.zeros(n_steps + 1)
    site_mean = np.zeros(n_steps + 1)
    purity_peak = np.zeros(n_steps + 1)
    trace_peak = np.zeros(n_steps + 1)
    peak_evals = []

    print(f"\nφ={mix_phi}, σ={sigma}, {n_steps} steps")
    print(f"{'t':>5} {'norm':>8} {'⟨d⟩_tet':>8} {'Var_tet':>9} "
          f"{'⟨d⟩_site':>9} {'Var_site':>9} {'peak tet':>9} {'purity':>8}")

    t0 = time.perf_counter()

    for t in range(n_steps + 1):
        # Site probabilities
        lib.manifold_site_probs(site_probs.ctypes.data_as(c_dp))

        # Site-level variance
        d1_s = np.sum(site_probs * site_dist)
        d2_s = np.sum(site_probs * site_dist**2)
        site_mean[t] = d1_s
        site_var[t] = d2_s - d1_s**2

        # Tet-level: aggregate probabilities by tet
        tet_probs = np.zeros(n_tets)
        for tet_id in range(n_tets):
            for s in tet_sites[tet_id]:
                tet_probs[tet_id] += site_probs[s]

        d1_t = np.sum(tet_probs * tet_dist)
        d2_t = np.sum(tet_probs * tet_dist**2)
        tet_mean[t] = d1_t
        tet_var[t] = d2_t - d1_t**2

        # Peak tet density matrix
        peak_tet = np.argmax(tet_probs)
        psi_re_now = np.zeros(dim, dtype=np.float64)
        psi_im_now = np.zeros(dim, dtype=np.float64)
        lib.manifold_get_psi(psi_re_now.ctypes.data_as(c_dp),
                             psi_im_now.ctypes.data_as(c_dp))
        psi_now = psi_re_now + 1j * psi_im_now

        rho = tet_reduced_density(psi_now, tet_sites[peak_tet])
        tr = np.real(np.trace(rho))
        pur = np.real(np.trace(rho @ rho)) / max(tr**2, 1e-30) if tr > 1e-15 else 0
        trace_peak[t] = tr
        purity_peak[t] = pur
        evals = np.sort(np.real(np.linalg.eigvalsh(rho)))[::-1]
        peak_evals.append(evals)

        if t in record_times:
            norm = np.sqrt(lib.manifold_norm2())
            print(f"{t:5d} {norm:8.6f} {d1_t:8.2f} {tet_var[t]:9.2f} "
                  f"{d1_s:9.2f} {site_var[t]:9.2f} {peak_tet:9d} {pur:8.4f}")

        if t < n_steps:
            lib.manifold_step(mix_phi)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.1f}s ({1000*elapsed/n_steps:.1f} ms/step)")

    # ================================================================
    # Fit Var(d) = a + b*t + c*t² (ballistic if c > 0)
    # ================================================================
    t_arr = np.arange(n_steps + 1).astype(float)
    fit_start = max(5, int(sigma))

    coeffs_tet = np.polyfit(t_arr[fit_start:], tet_var[fit_start:], 2)
    c_tet, b_tet, a_tet = coeffs_tet
    print(f"\nTet-level:  Var(d) ≈ {a_tet:.2f} + {b_tet:.4f}·t + {c_tet:.6f}·t²")
    if c_tet > 0:
        print(f"  → Ballistic speed (tet hops/step): v = √c = {np.sqrt(c_tet):.4f}")
    else:
        print(f"  → Sub-ballistic (c < 0)")

    coeffs_site = np.polyfit(t_arr[fit_start:], site_var[fit_start:], 2)
    c_site, b_site, a_site = coeffs_site
    print(f"Site-level: Var(d) ≈ {a_site:.2f} + {b_site:.4f}·t + {c_site:.6f}·t²")
    if c_site > 0:
        print(f"  → Ballistic speed (site hops/step): v = √c = {np.sqrt(c_site):.4f}")

    # ================================================================
    # Plot
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Variance vs time
    ax = axes[0, 0]
    ax.plot(t_arr, tet_var, 'b-', lw=1.5, label='Tet-level Var(d)')
    ax.plot(t_arr, site_var, 'r-', lw=1.5, alpha=0.5, label='Site-level Var(d)')
    t_fit = t_arr[fit_start:]
    ax.plot(t_fit, np.polyval(coeffs_tet, t_fit), 'b--', lw=1, alpha=0.7,
            label=f'Tet fit: c={c_tet:.4f}')
    ax.plot(t_fit, np.polyval(coeffs_site, t_fit), 'r--', lw=1, alpha=0.5,
            label=f'Site fit: c={c_site:.4f}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Var(d)')
    ax.set_title('(a) Spreading: Var(d) vs time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # (b) Tet probability profile at selected times
    ax = axes[0, 1]
    snap_times = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]
    snap_times = [t for t in snap_times if t <= n_steps]
    # Re-run to get snapshots
    lib.manifold_init_wavepacket(origin_site, sigma, 0.0, 0.0, 0.0)
    lib.manifold_get_psi(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))
    for sid in range(nsites):
        d = site_dist[sid]
        w = np.exp(-d**2 / (2 * sigma**2))
        psi_re[4*sid:4*sid+4] *= w
        psi_im[4*sid:4*sid+4] *= w
    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)
    lib.manifold_set_psi_bulk(psi_re.ctypes.data_as(c_dp), psi_im.ctypes.data_as(c_dp))

    snap_idx = 0
    for t in range(max(snap_times) + 1):
        if t == snap_times[snap_idx]:
            lib.manifold_site_probs(site_probs.ctypes.data_as(c_dp))
            tet_probs = np.zeros(n_tets)
            for tet_id in range(n_tets):
                for s in tet_sites[tet_id]:
                    tet_probs[tet_id] += site_probs[s]
            ax.scatter(tet_dist, tet_probs, s=15, alpha=0.6, label=f't={t}')
            snap_idx += 1
            if snap_idx >= len(snap_times):
                break
        if t < max(snap_times):
            lib.manifold_step(mix_phi)
    ax.set_xlabel('Tet distance from origin')
    ax.set_ylabel('p(tet)')
    ax.set_title('(b) Tet probability vs distance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # (c) Purity at peak tet
    ax = axes[1, 0]
    ax.plot(t_arr, purity_peak, 'g-', lw=1.5)
    ax.axhline(0.25, color='gray', ls='--', alpha=0.4, label='Maximally mixed (4D)')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4, label='Pure state')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Purity Tr(ρ²)/Tr(ρ)²')
    ax.set_title('(c) Purity at densest tet')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # (d) Eigenvalues of rho at peak tet
    ax = axes[1, 1]
    peak_evals_arr = np.array(peak_evals)
    for i in range(4):
        # Normalize eigenvalues by trace
        normalized = peak_evals_arr[:, i] / np.maximum(trace_peak, 1e-30)
        ax.plot(t_arr, normalized, lw=1.2, label=f'λ_{i+1}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Eigenvalue / Tr(ρ)')
    ax.set_title('(d) Density matrix eigenvalues at densest tet')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f'Tet-level spreading: {n_tets} tets, φ={mix_phi}, σ={sigma}',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = '/tmp/tet_spreading.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
