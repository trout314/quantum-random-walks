#!/usr/bin/env python3
"""
Compute the spectrum of the walk operator on a manifold.

Builds the walk matrix W by applying it to each basis vector,
then diagonalizes to find eigenvalues on the unit circle.
"""
import ctypes, numpy as np, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from triangulation import Triangulation


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
    lib.manifold_set_psi.argtypes = [ctypes.c_int, c_dp, c_dp]
    lib.manifold_get_psi.argtypes = [c_dp, c_dp]
    return lib


def main():
    tri_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else
        '~/Desktop/Discrete-Differential-Geometry/standard_triangulations/equilibrated_200.mfd')

    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

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

    dim = 4 * nsites
    print(f"Hilbert space dimension: {dim}")
    print(f"Building walk matrix ({dim}×{dim}), φ={mix_phi}...", flush=True)

    # Build walk matrix: W[:, col] = W * e_col
    W = np.zeros((dim, dim), dtype=complex)
    out_re = np.zeros(dim, dtype=np.float64)
    out_im = np.zeros(dim, dtype=np.float64)
    re4 = np.zeros(4, dtype=np.float64)
    im4 = np.zeros(4, dtype=np.float64)

    for col in range(dim):
        if col % 500 == 0:
            print(f"  column {col}/{dim}", flush=True)

        site = col // 4
        comp = col % 4

        lib.manifold_zero_psi()
        re4[:] = 0
        re4[comp] = 1.0
        im4[:] = 0
        lib.manifold_set_psi(site,
                             re4.ctypes.data_as(c_dp),
                             im4.ctypes.data_as(c_dp))

        lib.manifold_step(mix_phi)

        lib.manifold_get_psi(
            out_re.ctypes.data_as(c_dp),
            out_im.ctypes.data_as(c_dp))
        W[:, col] = out_re + 1j * out_im

    # W is complex-linear, so this gives the full complex walk matrix.

    print("Diagonalizing...", flush=True)
    eigenvalues = np.linalg.eigvals(W)

    mags = np.abs(eigenvalues)
    phases = np.angle(eigenvalues)
    print(f"|λ| range: [{mags.min():.6f}, {mags.max():.6f}]")
    print(f"Phase range: [{phases.min():.4f}, {phases.max():.4f}]")

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Unit circle
    ax = axes[0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.2)
    ax.scatter(eigenvalues.real, eigenvalues.imag, s=2, alpha=0.5, c='blue')
    ax.set_aspect('equal')
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title('Eigenvalues on Unit Circle')

    # Density of states
    ax = axes[1]
    ax.hist(phases, bins=100, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Phase θ')
    ax.set_ylabel('Count')
    ax.set_title('Density of States')

    # |λ| histogram
    ax = axes[2]
    ax.hist(mags, bins=50, color='green', edgecolor='darkgreen', alpha=0.7)
    ax.axvline(1.0, color='r', linestyle='--')
    ax.set_xlabel('|λ|')
    ax.set_ylabel('Count')
    ax.set_title('Unitarity Check')

    fig.suptitle(f'Walk Spectrum: {tri.n_tets} tets, {dim}D, φ={mix_phi}', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_spectrum.png'
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")

    np.save('/tmp/manifold_eigenvalues.npy', eigenvalues)
    print(f"Eigenvalues saved to /tmp/manifold_eigenvalues.npy")


if __name__ == '__main__':
    main()
