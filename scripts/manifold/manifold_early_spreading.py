#!/usr/bin/env python3
"""
Early-time spreading test on multiple manifold sizes.

Measures Var(d) for t=0..15 before finite-size saturation kicks in.
Compares ballistic (Var ~ t²) vs diffusive (Var ~ t) scaling.
Also varies φ to test mass dependence.
"""
import numpy as np
import ctypes, os, sys, time, glob
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from triangulation import Triangulation

SEEDS_DIR = os.path.expanduser('~/Desktop/Discrete-Differential-Geometry/seeds')


def load_d_library():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, '..', '..', 'dlang', 'build', 'quantum_walk.so')
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


def run_early_spreading(lib, tri_path, mix_phi, n_steps=15, sigma=3.0):
    """Run walk for n_steps, return (var_arr, d1_arr, norm_arr, nsites, diameter)."""
    tri = Triangulation.load(tri_path)
    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    neighbors_flat = np.array(tri.neighbors, dtype=np.int32).flatten()

    old = suppress_stdout()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        neighbors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tri.n_tets)
    nsites = lib.manifold_build_lattice()
    restore_stdout(old)

    adj = build_site_adjacency(lib, nsites)
    site_dist = bfs_distances_site(adj, 0, nsites)
    diameter = int(site_dist.max())

    # Frame-transported IC with graph-distance Gaussian
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

    # Measure
    probs = np.zeros(nsites, dtype=np.float64)
    var_arr = np.zeros(n_steps + 1)
    d1_arr = np.zeros(n_steps + 1)

    lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    d1 = np.sum(probs * site_dist_f)
    d2 = np.sum(probs * site_dist_f**2)
    var_arr[0] = d2 - d1**2
    d1_arr[0] = d1

    for t in range(1, n_steps + 1):
        lib.manifold_step(mix_phi)
        lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        d1 = np.sum(probs * site_dist_f)
        d2 = np.sum(probs * site_dist_f**2)
        var_arr[t] = d2 - d1**2
        d1_arr[t] = d1

    return var_arr, d1_arr, tri.n_tets, nsites, diameter


def main():
    lib = load_d_library()
    n_steps = 15

    # Test across manifold sizes with φ=0.05
    sizes = ['1778', '3162', '5623', '17783']
    phi_values = [0.05, 0.10, 0.20, 0.40]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Var(d) vs t for different manifold sizes, φ=0.05
    ax = axes[0, 0]
    size_cache = []  # (n_tets, diam, dvar)
    for size in sizes:
        path = find_seed(size)
        if path is None:
            continue
        t0 = time.perf_counter()
        var, d1, n_tets, nsites, diam = run_early_spreading(lib, path, 0.05, n_steps)
        elapsed = time.perf_counter() - t0
        dvar = var - var[0]
        size_cache.append((n_tets, diam, dvar))
        print(f"N={n_tets:6d} ({nsites:7d} sites, diam={diam:3d}): "
              f"ΔVar(15)={dvar[-1]:.2f}  [{elapsed:.1f}s]")
        t_arr = np.arange(n_steps + 1)
        ax.plot(t_arr, dvar, 'o-', markersize=3, label=f'{n_tets} tets (d={diam})')
    ax.set_xlabel('Step')
    ax.set_ylabel('ΔVar(d)')
    ax.set_title('Size dependence (φ=0.05)')
    ax.legend(fontsize=7)

    # Panel 2: ΔVar/t² for same (reuse cached data)
    ax = axes[0, 1]
    for n_tets, diam, dvar in size_cache:
        t_arr = np.arange(n_steps + 1)
        t_nz = t_arr[1:]
        ax.plot(t_nz, dvar[1:] / t_nz**2, 'o-', markersize=3, label=f'{n_tets} tets')
    ax.set_xlabel('Step')
    ax.set_ylabel('ΔVar(d)/t²')
    ax.set_title('Ballistic test (φ=0.05)')
    ax.legend(fontsize=7)

    # Panel 3: φ dependence on largest available
    ax = axes[1, 0]
    path = find_seed('17783')
    if path is None:
        path = find_seed('5623')
    phi_cache = []  # (phi, dvar)
    for phi in phi_values:
        t0 = time.perf_counter()
        var, d1, n_tets, nsites, diam = run_early_spreading(lib, path, phi, n_steps)
        elapsed = time.perf_counter() - t0
        dvar = var - var[0]
        phi_cache.append((phi, dvar))
        print(f"φ={phi:.2f}: ΔVar(15)={dvar[-1]:.2f}  [{elapsed:.1f}s]")
        t_arr = np.arange(n_steps + 1)
        ax.plot(t_arr, dvar, 'o-', markersize=3, label=f'φ={phi}')
    ax.set_xlabel('Step')
    ax.set_ylabel('ΔVar(d)')
    ax.set_title(f'φ dependence ({n_tets} tets)')
    ax.legend(fontsize=7)

    # Panel 4: ΔVar/t² for φ dependence (reuse cache)
    ax = axes[1, 1]
    for phi, dvar in phi_cache:
        t_arr = np.arange(n_steps + 1)
        t_nz = t_arr[1:]
        ax.plot(t_nz, dvar[1:] / t_nz**2, 'o-', markersize=3, label=f'φ={phi}')
    ax.set_xlabel('Step')
    ax.set_ylabel('ΔVar(d)/t²')
    ax.set_title(f'Ballistic test vs φ ({n_tets} tets)')
    ax.legend(fontsize=7)

    fig.suptitle('Early-time spreading: frame-transported balanced IC, σ=3', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_early_spreading.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
