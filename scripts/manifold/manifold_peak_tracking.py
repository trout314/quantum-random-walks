#!/usr/bin/env python3
"""
Track the wavefront (peak probability distance) vs time.

Ballistic: d_peak ~ t  (linear)
Diffusive: d_peak ~ √t (square root)

Also track the leading edge (distance where P(d) drops below threshold).
"""
import numpy as np
import ctypes, os, sys, time, glob
from collections import deque

from src.triangulation import Triangulation

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


def setup_ic(lib, nsites, site_dist, sigma):
    """Frame-transported balanced IC with graph-distance Gaussian."""
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


def radial_profile(probs, site_dist, max_dist):
    profile = np.zeros(max_dist + 1)
    for d in range(max_dist + 1):
        mask = site_dist == d
        profile[d] = probs[mask].sum()
    return profile


def density_profile(probs, site_dist, max_dist, shell_counts):
    """Probability density = P(d) / N(d) where N(d) is shell count."""
    profile = np.zeros(max_dist + 1)
    for d in range(max_dist + 1):
        mask = site_dist == d
        total = probs[mask].sum()
        if shell_counts[d] > 0:
            profile[d] = total / shell_counts[d]
    return profile


def run_walk_tracking(lib, tri_path, mix_phi, n_steps, sigma=3.0):
    """Run quantum + classical walk, tracking wavefront position."""
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
    max_dist = int(site_dist.max())
    shell_counts = np.array([np.sum(site_dist == d) for d in range(max_dist + 1)])

    # Quantum walk
    setup_ic(lib, nsites, site_dist, sigma)
    probs = np.zeros(nsites, dtype=np.float64)

    q_peak = np.zeros(n_steps + 1)
    q_mean = np.zeros(n_steps + 1)
    q_leading = np.zeros(n_steps + 1)  # distance where cumulative P > 0.99

    lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    rp = radial_profile(probs, site_dist, max_dist)
    dp = density_profile(probs, site_dist, max_dist, shell_counts)
    q_peak[0] = np.argmax(dp)
    q_mean[0] = np.sum(probs * site_dist.astype(float))
    cum = np.cumsum(rp)
    q_leading[0] = np.searchsorted(cum, 0.99)

    for t in range(1, n_steps + 1):
        lib.manifold_step(mix_phi)
        lib.manifold_site_probs(probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        rp = radial_profile(probs, site_dist, max_dist)
        dp = density_profile(probs, site_dist, max_dist, shell_counts)
        q_peak[t] = np.argmax(dp)
        q_mean[t] = np.sum(probs * site_dist.astype(float))
        cum = np.cumsum(rp)
        q_leading[t] = np.searchsorted(cum, 0.99)

    # Classical random walk
    rng = np.random.default_rng(42)
    n_trials = 200000
    site_dist_f = site_dist.astype(np.float64)
    ic_probs = np.exp(-site_dist_f**2 / sigma**2)
    ic_probs /= ic_probs.sum()
    positions = rng.choice(nsites, size=n_trials, p=ic_probs)

    c_mean = np.zeros(n_steps + 1)
    c_leading = np.zeros(n_steps + 1)

    hist = np.bincount(site_dist[positions], minlength=max_dist+1).astype(float) / n_trials
    c_mean[0] = np.sum(hist * np.arange(max_dist + 1))
    cum = np.cumsum(hist)
    c_leading[0] = np.searchsorted(cum, 0.99)

    for t in range(1, n_steps + 1):
        new_pos = np.empty_like(positions)
        for i in range(n_trials):
            nbs = adj[positions[i]]
            new_pos[i] = nbs[rng.integers(len(nbs))]
        positions = new_pos
        hist = np.bincount(site_dist[positions], minlength=max_dist+1).astype(float) / n_trials
        c_mean[t] = np.sum(hist * np.arange(max_dist + 1))
        cum = np.cumsum(hist)
        c_leading[t] = np.searchsorted(cum, 0.99)

    return {
        'n_tets': tri.n_tets, 'nsites': nsites, 'diameter': max_dist,
        'q_peak': q_peak, 'q_mean': q_mean, 'q_leading': q_leading,
        'c_mean': c_mean, 'c_leading': c_leading,
    }


def main():
    lib = load_d_library()
    n_steps = 20
    mix_phi = 0.05

    path = find_seed('17783')
    print(f"Running on {path}...")

    t0 = time.perf_counter()
    r = run_walk_tracking(lib, path, mix_phi, n_steps)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"Manifold: {r['n_tets']} tets, {r['nsites']} sites, diameter={r['diameter']}")

    t_arr = np.arange(n_steps + 1)
    print(f"\n{'t':>3} {'q_peak':>7} {'q_mean':>7} {'q_lead':>7} {'c_mean':>7} {'c_lead':>7}")
    for t in range(n_steps + 1):
        print(f"{t:3d} {r['q_peak'][t]:7.1f} {r['q_mean'][t]:7.2f} {r['q_leading'][t]:7.1f} "
              f"{r['c_mean'][t]:7.2f} {r['c_leading'][t]:7.1f}")

    # Fit mean distance: quantum d ~ a + v*t (ballistic), classical d ~ a + b*√t
    # Use t >= 2 to avoid transient
    fit_q = np.polyfit(t_arr[2:], r['q_mean'][2:], 1)
    fit_c_sqrt = np.polyfit(np.sqrt(t_arr[2:]), r['c_mean'][2:], 1)
    print(f"\nQuantum ⟨d⟩ ≈ {fit_q[1]:.2f} + {fit_q[0]:.4f}·t  (v = {fit_q[0]:.4f} sites/step)")
    print(f"Classical ⟨d⟩ ≈ {fit_c_sqrt[1]:.2f} + {fit_c_sqrt[0]:.4f}·√t")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(t_arr, r['q_mean'], 'b-o', markersize=3, label='Quantum ⟨d⟩')
    ax.plot(t_arr, r['c_mean'], 'r-s', markersize=3, label='Classical ⟨d⟩')
    ax.plot(t_arr, fit_q[1] + fit_q[0] * t_arr, 'b--', alpha=0.5, label=f'v={fit_q[0]:.3f}')
    ax.plot(t_arr, fit_c_sqrt[1] + fit_c_sqrt[0] * np.sqrt(t_arr), 'r--', alpha=0.5)
    ax.axhline(r['diameter'], color='gray', linestyle=':', alpha=0.5, label=f'diameter={r["diameter"]}')
    ax.set_xlabel('Step')
    ax.set_ylabel('⟨d⟩')
    ax.set_title('Mean distance from origin')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(t_arr, r['q_leading'], 'b-o', markersize=3, label='Quantum (99%)')
    ax.plot(t_arr, r['c_leading'], 'r-s', markersize=3, label='Classical (99%)')
    ax.axhline(r['diameter'], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Leading edge d')
    ax.set_title('Wavefront position (99% cumulative)')
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(t_arr[1:], r['q_mean'][1:] / t_arr[1:], 'b-o', markersize=3, label='Quantum ⟨d⟩/t')
    ax.plot(t_arr[1:], r['c_mean'][1:] / t_arr[1:], 'r-s', markersize=3, label='Classical ⟨d⟩/t')
    ax.set_xlabel('Step')
    ax.set_ylabel('⟨d⟩/t')
    ax.set_title('Velocity (plateau = ballistic)')
    ax.legend(fontsize=7)

    fig.suptitle(f'Peak tracking: {r["n_tets"]} tets, φ={mix_phi}, σ=3', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_peak_tracking.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
