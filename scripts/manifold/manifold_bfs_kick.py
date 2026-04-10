#!/usr/bin/env python3
"""
Wavepacket kick test on 100k manifold, measured via BFS distance profile.
Compares k=0 baseline with several kick values.
"""
import ctypes, numpy as np, os, sys
from collections import deque

from src.triangulation import Triangulation


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
    print(f"Lattice: {nsites} sites, {lib.manifold_nchains()} chains")

    # BFS distances from origin tet
    print("Computing BFS distances from tet 0...")
    tet_dist = bfs_distances(tri.neighbors, 0)
    max_dist = tet_dist.max()
    print(f"  Dual graph: max BFS distance = {max_dist}")

    # Site -> tet mapping
    site_tets = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(site_tets.ctypes.data_as(c_ip))
    site_dist = tet_dist[site_tets]

    # Parameters
    mix_phi = 0.05
    sigma = 30.0
    n_steps = 100
    snapshot_steps = [0, 10, 20, 40, 60, 80, 100]
    k_values = [0.0, 0.05, 0.10, 0.20]

    print(f"\nσ={sigma}, φ={mix_phi}, {n_steps} steps")
    print(f"m ≈ {0.878*mix_phi:.4f}, 1/m ≈ {1/(0.878*mix_phi):.1f}")
    print(f"k values: {k_values}")

    site_probs = np.zeros(nsites, dtype=np.float64)

    def get_bfs_profile():
        lib.manifold_site_probs(site_probs.ctypes.data_as(c_dp))
        prob_by_dist = np.zeros(max_dist + 1)
        for d in range(max_dist + 1):
            mask = site_dist == d
            prob_by_dist[d] = site_probs[mask].sum()
        return prob_by_dist

    def mean_and_std(profile):
        d_arr = np.arange(len(profile))
        mu = np.average(d_arr, weights=profile + 1e-30)
        var = np.average((d_arr - mu)**2, weights=profile + 1e-30)
        return mu, np.sqrt(var)

    # Run each k value
    all_profiles = {}  # k -> {t -> profile}
    all_stats = {}     # k -> {t -> (mean, std)}

    for k in k_values:
        print(f"\n--- k = {k:.3f} ---")
        lib.manifold_init_wavepacket(0, sigma, k, 0.0, 0.0)
        norm0 = np.sqrt(lib.manifold_norm2())
        print(f"  t=0 norm={norm0:.9f}")

        profiles = {}
        stats = {}

        # t=0 snapshot
        profiles[0] = get_bfs_profile()
        stats[0] = mean_and_std(profiles[0])
        print(f"  t=  0  <d>={stats[0][0]:.1f}  σ_d={stats[0][1]:.1f}", flush=True)

        t = 0
        for target in snapshot_steps[1:]:
            while t < target:
                lib.manifold_step(mix_phi)
                t += 1
            profiles[t] = get_bfs_profile()
            stats[t] = mean_and_std(profiles[t])
            norm = np.sqrt(lib.manifold_norm2())
            print(f"  t={t:3d}  <d>={stats[t][0]:.1f}  σ_d={stats[t][1]:.1f}  "
                  f"norm={norm:.9f}", flush=True)

        all_profiles[k] = profiles
        all_stats[k] = stats

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))

    # P(d) at final time for each k
    ax = axes[0, 0]
    for i, k in enumerate(k_values):
        p = all_profiles[k][n_steps]
        ax.plot(np.arange(len(p)), p, color=colors[i], label=f'k={k:.2f}')
    ax.set_xlabel('BFS distance from origin tet')
    ax.set_ylabel('Total probability')
    ax.set_title(f'P(d) at t={n_steps}')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 80)

    # P(d) difference from k=0 at final time
    ax = axes[0, 1]
    p0 = all_profiles[0.0][n_steps]
    for i, k in enumerate(k_values[1:], 1):
        diff = all_profiles[k][n_steps] - p0
        ax.plot(np.arange(len(diff)), diff, color=colors[i], label=f'k={k:.2f}')
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('BFS distance from origin tet')
    ax.set_ylabel('ΔP(d) vs k=0')
    ax.set_title(f'Kick effect on P(d) at t={n_steps}')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 80)

    # <d> vs time for each k
    ax = axes[1, 0]
    for i, k in enumerate(k_values):
        ts = sorted(all_stats[k].keys())
        means = [all_stats[k][t][0] for t in ts]
        ax.plot(ts, means, 'o-', color=colors[i], markersize=4, label=f'k={k:.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('⟨d⟩ (BFS distance)')
    ax.set_title('Mean BFS distance vs time')
    ax.legend(fontsize=8)

    # <d> difference from k=0
    ax = axes[1, 1]
    ts = sorted(all_stats[0.0].keys())
    base_means = [all_stats[0.0][t][0] for t in ts]
    for i, k in enumerate(k_values[1:], 1):
        means = [all_stats[k][t][0] for t in ts]
        delta = [m - b for m, b in zip(means, base_means)]
        ax.plot(ts, delta, 'o-', color=colors[i], markersize=4, label=f'k={k:.2f}')
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Δ⟨d⟩ vs k=0')
    ax.set_title('Kick effect on mean BFS distance')
    ax.legend(fontsize=8)

    fig.suptitle(f'Manifold Kick (BFS): {tri.n_tets} tets, {nsites} sites, '
                 f'σ={sigma:.0f}, φ={mix_phi}', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_bfs_kick.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
