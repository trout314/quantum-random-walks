#!/usr/bin/env python3
"""
Radial probability profile P(d) vs graph distance d at different times.

For Dirac-like behavior: P(d,t) should show a shell/wavefront at d ~ v*t
(ballistic, ring-like), not a Gaussian centered at d=0 (diffusive).
"""
import numpy as np
import ctypes, os, sys, time, glob
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from triangulation import Triangulation

SEEDS_DIR = os.path.expanduser('~/Desktop/Discrete-Differential-Geometry/seeds')


def load_d_library():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, '..', 'dlang', 'build', 'quantum_walk.so')
    lib = ctypes.CDLL(lib_path)
    lib.manifold_create.argtypes = [ctypes.c_int]
    lib.manifold_create.restype = None
    lib.manifold_nsites.argtypes = []
    lib.manifold_nsites.restype = ctypes.c_int
    lib.manifold_nchains.argtypes = []
    lib.manifold_nchains.restype = ctypes.c_int
    lib.manifold_load_triangulation.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.manifold_load_triangulation.restype = None
    lib.manifold_build_lattice.argtypes = []
    lib.manifold_build_lattice.restype = ctypes.c_int
    lib.manifold_set_psi.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.manifold_set_psi.restype = None
    lib.manifold_norm2.argtypes = []
    lib.manifold_norm2.restype = ctypes.c_double
    lib.manifold_step.argtypes = [ctypes.c_double]
    lib.manifold_step.restype = ctypes.c_double
    lib.manifold_init_wavepacket.argtypes = [
        ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.manifold_init_wavepacket.restype = None
    lib.manifold_site_probs.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.manifold_site_probs.restype = ctypes.c_double
    lib.manifold_get_psi.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.manifold_get_psi.restype = None
    lib.manifold_chain_next.argtypes = [ctypes.c_int, ctypes.c_bool]
    lib.manifold_chain_next.restype = ctypes.c_int
    lib.manifold_chain_prev.argtypes = [ctypes.c_int, ctypes.c_bool]
    lib.manifold_chain_prev.restype = ctypes.c_int
    lib.manifold_has_chain.argtypes = [ctypes.c_int, ctypes.c_bool]
    lib.manifold_has_chain.restype = ctypes.c_bool
    return lib


def suppress_stdout():
    sys.stdout.flush()
    old = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return old

def restore_stdout(old):
    os.dup2(old, 1)
    os.close(old)


def find_seed(target_n):
    pattern = os.path.join(SEEDS_DIR, f'S3_N{target_n}_1e-1_ED5p0043_1e-1_s000.mfd')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    pattern = os.path.join(SEEDS_DIR, f'S3_N{target_n}_*_s000.mfd')
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def build_site_adjacency(lib, nsites):
    adj = [[] for _ in range(nsites)]
    for sid in range(nsites):
        for isR in [True, False]:
            if not lib.manifold_has_chain(sid, isR):
                continue
            nxt = lib.manifold_chain_next(sid, isR)
            prv = lib.manifold_chain_prev(sid, isR)
            if nxt >= 0 and nxt not in adj[sid]:
                adj[sid].append(nxt)
            if prv >= 0 and prv not in adj[sid]:
                adj[sid].append(prv)
    return adj


def bfs_distances_site(adj, origin, nsites):
    dist = np.full(nsites, -1, dtype=np.int32)
    dist[origin] = 0
    queue = deque([origin])
    while queue:
        s = queue.popleft()
        for nb in adj[s]:
            if dist[nb] < 0:
                dist[nb] = dist[s] + 1
                queue.append(nb)
    return dist


def radial_profile(probs, site_dist, max_dist):
    """Sum probability in each distance shell."""
    profile = np.zeros(max_dist + 1)
    for d in range(max_dist + 1):
        mask = site_dist == d
        profile[d] = probs[mask].sum()
    return profile


def main():
    tri_path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else
        find_seed('17783'))
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    tri = Triangulation.load(tri_path)
    print(f"Manifold: {tri.n_tets} tets, {tri.n_verts} verts")

    lib = load_d_library()
    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    neighbors_flat = np.array(tri.neighbors, dtype=np.int32).flatten()

    old = suppress_stdout()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        neighbors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tri.n_tets)
    nsites = lib.manifold_build_lattice()
    restore_stdout(old)
    print(f"Lattice: {nsites} sites, {lib.manifold_nchains()} chains")

    adj = build_site_adjacency(lib, nsites)
    site_dist = bfs_distances_site(adj, 0, nsites)
    max_dist = int(site_dist.max())
    print(f"Site diameter: {max_dist}")

    # Count sites per distance shell
    shell_counts = np.array([np.sum(site_dist == d) for d in range(max_dist + 1)])
    print("Shell counts:", shell_counts[:15], "...")

    # Frame-transported IC with graph-distance Gaussian
    sigma = 3.0
    lib.manifold_init_wavepacket(0, 1e10, 0.0, 0.0, 0.0)
    psi_re = np.zeros(4 * nsites, dtype=np.float64)
    psi_im = np.zeros(4 * nsites, dtype=np.float64)
    lib.manifold_get_psi(
        psi_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        psi_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    site_dist_f = site_dist.astype(np.float64)
    for sid in range(nsites):
        w = np.exp(-site_dist_f[sid]**2 / (2 * sigma**2))
        psi_re[4*sid:4*sid+4] *= w
        psi_im[4*sid:4*sid+4] *= w

    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)

    for sid in range(nsites):
        lib.manifold_set_psi(sid,
            psi_re[4*sid:4*sid+4].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            psi_im[4*sid:4*sid+4].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # Collect radial profiles at selected times
    snapshot_times = [0, 2, 5, 8, 12, 18, 25]
    n_steps = max(snapshot_times)
    profiles = {}

    probs = np.zeros(nsites, dtype=np.float64)
    lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    profiles[0] = radial_profile(probs, site_dist, max_dist)

    t0 = time.perf_counter()
    for t in range(1, n_steps + 1):
        lib.manifold_step(mix_phi)
        if t in snapshot_times:
            lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            profiles[t] = radial_profile(probs, site_dist, max_dist)
            elapsed = time.perf_counter() - t0
            peak_d = np.argmax(profiles[t])
            print(f"  t={t:3d}: peak at d={peak_d}, P(peak)={profiles[t][peak_d]:.4f}  [{elapsed:.1f}s]")

    # Also run a classical random walk for comparison
    # (uniform hop to random neighbor at each step)
    print("\nClassical random walk comparison...")
    n_trials = 100000
    rng = np.random.default_rng(42)
    # Start from same IC distribution (Gaussian in distance)
    # Sample starting positions according to IC
    ic_probs = np.zeros(nsites)
    for sid in range(nsites):
        w = np.exp(-site_dist_f[sid]**2 / (2 * sigma**2))
        ic_probs[sid] = w**2  # prob ~ |ψ|² ~ w²
    ic_probs /= ic_probs.sum()
    positions = rng.choice(nsites, size=n_trials, p=ic_probs)

    classical_profiles = {}
    for t_snap in snapshot_times:
        classical_profiles[t_snap] = None

    # Step the random walk
    classical_profiles[0] = radial_profile(
        np.bincount(site_dist[positions], minlength=max_dist+1).astype(float) / n_trials,
        np.arange(max_dist + 1), max_dist)

    for t in range(1, n_steps + 1):
        new_pos = np.empty_like(positions)
        for i in range(n_trials):
            neighbors = adj[positions[i]]
            new_pos[i] = neighbors[rng.integers(len(neighbors))]
        positions = new_pos
        if t in snapshot_times:
            hist = np.bincount(site_dist[positions], minlength=max_dist+1).astype(float) / n_trials
            classical_profiles[t] = hist

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_panels = len(snapshot_times)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    d_arr = np.arange(max_dist + 1)
    # Normalize by shell volume (number of sites at distance d)
    # to get density per site rather than total prob in shell
    for idx, t in enumerate(snapshot_times):
        ax = axes[idx]
        qw = profiles[t]
        cw = classical_profiles[t]

        # Raw probability per shell
        ax.bar(d_arr - 0.15, qw, width=0.3, color='steelblue', alpha=0.8, label='Quantum')
        ax.bar(d_arr + 0.15, cw, width=0.3, color='orange', alpha=0.8, label='Classical')

        ax.set_xlabel('Graph distance d')
        ax.set_ylabel('P(d)')
        ax.set_title(f't = {t}')
        if idx == 0:
            ax.legend(fontsize=7)
        ax.set_xlim(-0.5, max_dist + 0.5)

    # Hide unused
    for idx in range(len(snapshot_times), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Radial Profile: {tri.n_tets} tets, φ={mix_phi}, σ={sigma}', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_radial_profile.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
