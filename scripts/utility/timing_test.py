#!/usr/bin/env python3
"""Timing test: measure lattice build + walk step times across manifold sizes."""
import numpy as np
import ctypes, os, sys, time, glob

from src.triangulation import Triangulation

SEEDS_DIR = os.path.expanduser('~/Desktop/Discrete-Differential-Geometry/seeds')

SIZES = ['1e2', '178', '316', '562', '1778', '3162', '5623', '17783', '31623']


def find_seed(target_n):
    pattern = os.path.join(SEEDS_DIR, f'S3_N{target_n}_1e-1_ED5p0043_1e-1_s000.mfd')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    pattern = os.path.join(SEEDS_DIR, f'S3_N{target_n}_*_s000.mfd')
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def suppress_stdout():
    """Redirect C-level stdout to /dev/null, return restore function."""
    sys.stdout.flush()
    old = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return old

def restore_stdout(old):
    os.dup2(old, 1)
    os.close(old)


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

    return lib


def run_timing(lib, tri_path, n_steps=200, mix_phi=0.05):
    tri = Triangulation.load(tri_path)

    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    neighbors_flat = np.array(tri.neighbors, dtype=np.int32).flatten()

    # Time lattice build (suppress D's verbose chain tracing output)
    old = suppress_stdout()
    t0 = time.perf_counter()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        neighbors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tri.n_tets)
    nsites = lib.manifold_build_lattice()
    t_build = time.perf_counter() - t0
    restore_stdout(old)

    nchains = lib.manifold_nchains()

    # Set IC
    ic_re = np.array([1/np.sqrt(2), 0, 0, 0], dtype=np.float64)
    ic_im = np.array([0, 0, 1/np.sqrt(2), 0], dtype=np.float64)
    lib.manifold_set_psi(0,
                         ic_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         ic_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # Time walk steps
    t0 = time.perf_counter()
    for _ in range(n_steps):
        lib.manifold_step(mix_phi)
    t_walk = time.perf_counter() - t0

    norm = np.sqrt(lib.manifold_norm2())

    return {
        'n_tets': tri.n_tets,
        'n_sites': nsites,
        'n_chains': nchains,
        'build_s': t_build,
        'walk_s': t_walk,
        'step_ms': 1000 * t_walk / n_steps,
        'norm': norm,
        'n_steps': n_steps,
    }


def main():
    lib = load_d_library()

    print(f"{'tets':>8} {'sites':>8} {'chains':>7} {'build(s)':>9} "
          f"{'walk(s)':>9} {'ms/step':>9} {'norm':>14}")
    print('-' * 78)

    for size in SIZES:
        path = find_seed(size)
        if path is None:
            print(f"  No seed found for N={size}")
            continue
        n_steps = 200 if int(float(size)) < 20000 else 50
        r = run_timing(lib, path, n_steps=n_steps)
        print(f"{r['n_tets']:8d} {r['n_sites']:8d} {r['n_chains']:7d} "
              f"{r['build_s']:9.3f} {r['walk_s']:9.3f} {r['step_ms']:9.3f} "
              f"{r['norm']:14.12f}  ({r['n_steps']} steps)")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
