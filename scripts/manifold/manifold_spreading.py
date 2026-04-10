#!/usr/bin/env python3
"""
Manifold walk with frame-transported balanced IC.

Uses chain-network distance (site-level BFS along R/L chain links)
to measure spreading on the 12-fold cover lattice.
"""
import numpy as np
import ctypes, os, sys, time
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
        ctypes.c_int, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.manifold_init_wavepacket.restype = None
    lib.manifold_site_probs.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.manifold_site_probs.restype = ctypes.c_double
    lib.manifold_get_psi.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.manifold_get_psi.restype = None
    lib.manifold_get_site_tets.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.manifold_get_site_tets.restype = None

    # Chain query functions
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


def build_site_adjacency(lib, nsites):
    """Build adjacency list from chain links: each site connects to its
    R-chain next/prev and L-chain next/prev."""
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
    """BFS on site adjacency graph from origin."""
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


def main():
    tri_path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else
        os.path.join(SEEDS_DIR, 'S3_N5623_1e-1_ED5p0043_1e-1_s000.mfd'))
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    mix_phi = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

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

    nchains = lib.manifold_nchains()
    print(f"Lattice: {nsites} sites, {nchains} chains")

    # Build site-level adjacency from chain links
    print("Building site adjacency graph...")
    t0 = time.perf_counter()
    adj = build_site_adjacency(lib, nsites)
    avg_deg = np.mean([len(a) for a in adj])
    print(f"  Done in {time.perf_counter()-t0:.1f}s, avg degree = {avg_deg:.1f}")

    # BFS from site 0
    print("Computing site-level BFS distances...")
    t0 = time.perf_counter()
    site_dist = bfs_distances_site(adj, 0, nsites)
    max_dist = site_dist.max()
    print(f"  Site-level diameter from origin: {max_dist} (in {time.perf_counter()-t0:.1f}s)")

    # Frame-transported balanced IC with Gaussian in site distance
    sigma = max(3.0, max_dist * 0.10)
    print(f"Using site-distance Gaussian σ = {sigma:.1f} (diameter={max_dist})")

    # Get frame-transported spinor via initWavepacket with huge sigma
    lib.manifold_init_wavepacket(0, 1e10, 0.0, 0.0, 0.0)

    # Read, modulate by site-distance Gaussian, write back
    psi_re = np.zeros(4 * nsites, dtype=np.float64)
    psi_im = np.zeros(4 * nsites, dtype=np.float64)
    lib.manifold_get_psi(
        psi_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        psi_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    for sid in range(nsites):
        d = site_dist[sid]
        w = np.exp(-d**2 / (2 * sigma**2))
        psi_re[4*sid:4*sid+4] *= w
        psi_im[4*sid:4*sid+4] *= w

    norm2 = np.sum(psi_re**2 + psi_im**2)
    psi_re /= np.sqrt(norm2)
    psi_im /= np.sqrt(norm2)

    for sid in range(nsites):
        lib.manifold_set_psi(sid,
            psi_re[4*sid:4*sid+4].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            psi_im[4*sid:4*sid+4].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    init_norm = np.sqrt(lib.manifold_norm2())
    print(f"Initial norm: {init_norm:.12f}")

    # Initial observables
    probs = np.zeros(nsites, dtype=np.float64)
    lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    d1_0 = np.sum(probs * site_dist)
    d2_0 = np.sum(probs * site_dist**2)
    var_0 = d2_0 - d1_0**2
    print(f"Initial: ⟨d⟩={d1_0:.2f}, ⟨d²⟩={d2_0:.2f}, Var(d)={var_0:.2f}")

    # Run walk
    print(f"\nWalk: {n_steps} steps, φ={mix_phi}")
    d2_arr = np.zeros(n_steps + 1)
    d1_arr = np.zeros(n_steps + 1)
    var_arr = np.zeros(n_steps + 1)
    norm_arr = np.zeros(n_steps + 1)
    d2_arr[0] = d2_0
    d1_arr[0] = d1_0
    var_arr[0] = var_0
    norm_arr[0] = init_norm**2

    t0 = time.perf_counter()
    print(f"{'t':>5} {'norm':>14} {'⟨d⟩':>8} {'Var(d)':>10} {'ΔVar':>10} {'ΔVar/t²':>10}")
    print(f"{'0':>5} {init_norm:14.12f} {d1_0:8.2f} {var_0:10.2f} {'—':>10} {'—':>10}")

    for t in range(1, n_steps + 1):
        lib.manifold_step(mix_phi)
        norm2 = lib.manifold_norm2()
        norm_arr[t] = norm2

        lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        d1 = np.sum(probs * site_dist)
        d2 = np.sum(probs * site_dist.astype(np.float64)**2)
        var = d2 - d1**2
        d2_arr[t] = d2
        d1_arr[t] = d1
        var_arr[t] = var

        if t <= 10 or t % 25 == 0 or t == n_steps:
            elapsed = time.perf_counter() - t0
            dvar = var - var_0
            ratio = dvar / t**2 if t > 0 else 0
            print(f"{t:5d} {np.sqrt(norm2):14.12f} {d1:8.2f} {var:10.2f} "
                  f"{dvar:10.2f} {ratio:10.6f}  [{elapsed:.1f}s]")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({1000*elapsed/n_steps:.1f} ms/step)")

    # Fit Var(d) = a + b*t + c*t²
    t_arr = np.arange(n_steps + 1)
    fit_start = 5
    coeffs = np.polyfit(t_arr[fit_start:], var_arr[fit_start:], 2)
    c_ball, b_diff, a_const = coeffs
    print(f"\nVar(d) fit: {c_ball:.6e}·t² + {b_diff:.6e}·t + {a_const:.2f}")
    if abs(c_ball) > 0.001:
        v_eff = np.sqrt(abs(c_ball))
        print(f"  → Ballistic component: v ≈ {v_eff:.4f} chain-links/step")
    if abs(b_diff) > 0.001:
        print(f"  → Diffusive component: D ≈ {b_diff/2:.4f} links²/step")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(t_arr, var_arr, 'b-', linewidth=1, label='Walk')
    ax.plot(t_arr[fit_start:], np.polyval(coeffs, t_arr[fit_start:]),
            'r--', linewidth=1, label=f'Fit')
    ax.set_xlabel('Step')
    ax.set_ylabel('Var(d)')
    ax.set_title('Site-Distance Variance')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    t_nz = t_arr[1:]
    dvar = var_arr[1:] - var_0
    ax.plot(t_nz, dvar / t_nz**2, 'b-', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('ΔVar(d)/t²')
    ax.set_title('Ballistic test (plateau = ballistic)')

    ax = axes[1, 0]
    ax.plot(t_arr, d1_arr, 'm-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('⟨d⟩')
    ax.set_title('Mean Site Distance')

    ax = axes[1, 1]
    ax.plot(t_arr, np.sqrt(norm_arr), 'g-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Unitarity')

    fig.suptitle(f'Manifold Walk: {tri.n_tets} tets, {nsites} sites, φ={mix_phi}, '
                 f'σ={sigma:.1f}, site-diameter={max_dist}',
                 fontsize=11)
    plt.tight_layout()
    out_path = '/tmp/manifold_spreading.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
