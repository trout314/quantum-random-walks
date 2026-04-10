#!/usr/bin/env python3
"""
Wavepacket on 100k manifold: measure probability vs BFS distance
from origin tet in the dual graph at several time snapshots.
"""
import ctypes, numpy as np, os, sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from triangulation import Triangulation


def load_lib():
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'dlang', 'build', 'quantum_walk.so')
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
    lib.manifold_init_wavepacket.argtypes = [
        ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.manifold_site_probs.argtypes = [c_dp]
    lib.manifold_site_probs.restype = ctypes.c_double
    lib.manifold_get_site_tets.argtypes = [c_ip]
    return lib


def bfs_distances(neighbors, start):
    """BFS from start tet, return distance array."""
    n = len(neighbors)
    dist = np.full(n, -1, dtype=np.int32)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def main():
    tri_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else
        '~/Desktop/Discrete-Differential-Geometry/data/vdv_ramp/3d_N100006_vdv17936.6203.mfd')
    tri = Triangulation.load(tri_path)
    print(f"Manifold: {tri.n_tets} tets, {tri.n_verts} verts")

    lib = load_lib()
    c_dp = ctypes.POINTER(ctypes.c_double)
    c_ip = ctypes.POINTER(ctypes.c_int)

    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    nbrs_flat = np.array(tri.neighbors, dtype=np.int32).flatten()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(c_ip), nbrs_flat.ctypes.data_as(c_ip), tri.n_tets)
    nsites = lib.manifold_build_lattice()
    nchains = lib.manifold_nchains()
    print(f"Lattice: {nsites} sites, {nchains} chains")

    # BFS distances from origin tet (tet 0)
    print("Computing BFS distances from tet 0...")
    tet_dist = bfs_distances(tri.neighbors, 0)
    max_dist = tet_dist.max()
    print(f"  Dual graph: max BFS distance = {max_dist}")

    # Site → tet mapping
    site_tets = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(site_tets.ctypes.data_as(c_ip))

    # Site BFS distance = tet BFS distance of its tet
    site_dist = tet_dist[site_tets]

    # Parameters
    mix_phi = 0.05
    sigma = 30.0
    n_steps = 150
    snapshot_steps = [0, 5, 10, 20, 40, 60, 80, 100, 120, 150]

    print(f"\nσ={sigma}, φ={mix_phi}, k=0 (no kick)")
    print(f"Snapshots at steps: {snapshot_steps}")

    # Init wavepacket: sigma=30, k=0
    lib.manifold_init_wavepacket(0, sigma, 0.0, 0.0, 0.0)
    norm0 = np.sqrt(lib.manifold_norm2())
    print(f"  t=0 norm={norm0:.9f}")

    # Collect probability profiles at snapshots
    profiles = {}
    site_probs = np.zeros(nsites, dtype=np.float64)

    def snapshot(t):
        lib.manifold_site_probs(site_probs.ctypes.data_as(c_dp))
        # Bin by BFS distance
        prob_by_dist = np.zeros(max_dist + 1)
        for d in range(max_dist + 1):
            mask = site_dist == d
            prob_by_dist[d] = site_probs[mask].sum()
        profiles[t] = prob_by_dist
        # Also compute some stats
        mean_d = np.average(np.arange(max_dist + 1), weights=prob_by_dist + 1e-30)
        norm = np.sqrt(lib.manifold_norm2())
        print(f"  t={t:3d}  norm={norm:.9f}  <d>={mean_d:.1f}  "
              f"P(d<5)={prob_by_dist[:5].sum():.4f}  "
              f"P(d<20)={prob_by_dist[:20].sum():.4f}  "
              f"P(d<50)={prob_by_dist[:50].sum():.4f}", flush=True)

    snapshot(0)

    t = 0
    for target in snapshot_steps[1:]:
        while t < target:
            lib.manifold_step(mix_phi)
            t += 1
        snapshot(t)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Probability vs BFS distance for each snapshot
    ax = axes[0, 0]
    for t in snapshot_steps:
        if t in profiles:
            ax.plot(np.arange(len(profiles[t])), profiles[t], label=f't={t}')
    ax.set_xlabel('BFS distance from origin tet')
    ax.set_ylabel('Total probability')
    ax.set_title('P(d) vs BFS distance')
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(max_dist, 120))

    # Same but log scale
    ax = axes[0, 1]
    for t in snapshot_steps:
        if t in profiles:
            p = profiles[t]
            ax.semilogy(np.arange(len(p)), p + 1e-20, label=f't={t}')
    ax.set_xlabel('BFS distance from origin tet')
    ax.set_ylabel('Total probability (log)')
    ax.set_title('P(d) vs BFS distance (log scale)')
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(max_dist, 120))

    # Cumulative probability vs BFS distance
    ax = axes[1, 0]
    for t in snapshot_steps:
        if t in profiles:
            cum = np.cumsum(profiles[t])
            ax.plot(np.arange(len(cum)), cum, label=f't={t}')
    ax.axhline(1.0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('BFS distance from origin tet')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('CDF: P(distance ≤ d)')
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(max_dist, 120))

    # Mean BFS distance vs time
    ax = axes[1, 1]
    ts = sorted(profiles.keys())
    mean_ds = []
    std_ds = []
    for t in ts:
        p = profiles[t]
        d_arr = np.arange(len(p))
        mu = np.average(d_arr, weights=p + 1e-30)
        var = np.average((d_arr - mu)**2, weights=p + 1e-30)
        mean_ds.append(mu)
        std_ds.append(np.sqrt(var))
    ax.errorbar(ts, mean_ds, yerr=std_ds, fmt='ko-', markersize=5, capsize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('BFS distance')
    ax.set_title('⟨d⟩ ± σ_d vs time')

    fig.suptitle(f'Manifold Walk: {tri.n_tets} tets, {nsites} sites, '
                 f'σ={sigma:.0f}, φ={mix_phi}, k=0', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_bfs_profile.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
