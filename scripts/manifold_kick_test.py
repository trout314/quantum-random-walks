#!/usr/bin/env python3
"""
Wavepacket kick test on a triangulated 3-manifold.

Uses the same IC as the 3D flat-space walk: BFS frame-transport of
balanced spinor (1,0,i,0)/√2, Gaussian envelope, momentum kick.
Tracks ⟨x⟩, ⟨y⟩, ⟨z⟩ vs time to measure group velocity.
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
    lib.manifold_get_site_positions.argtypes = [c_dp]
    lib.manifold_kick_run_3d.argtypes = [
        ctypes.c_int,  # originSite
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # sigma, kx, ky, kz
        ctypes.c_double, ctypes.c_int, c_dp]  # mixPhi, nSteps, out
    lib.manifold_mean_position.argtypes = [c_dp]
    return lib


def main():
    tri_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else
        '~/Desktop/Discrete-Differential-Geometry/standard_triangulations/equilibrated_2000.mfd')
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

    # Get site positions
    pos = np.zeros(3 * nsites, dtype=np.float64)
    lib.manifold_get_site_positions(pos.ctypes.data_as(c_dp))
    pos = pos.reshape(nsites, 3)

    r_from_origin = np.sqrt((pos**2).sum(axis=1))
    print(f"Position range: x=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}]")
    print(f"Max |r| from site 0: {r_from_origin.max():.1f}")

    # Walk parameters
    mix_phi = 0.05
    n_steps = 60
    sigma = r_from_origin.max() / 6

    # Kick along x-axis with various momenta
    k_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    print(f"\nσ={sigma:.1f}, φ_mix={mix_phi}, {n_steps} steps")
    print(f"Origin site: 0 at ({pos[0,0]:.1f}, {pos[0,1]:.1f}, {pos[0,2]:.1f})")
    print(f"{'k':>6} {'v_x':>8} {'v_y':>8} {'v_z':>8} {'|v|':>8}")

    results = {}
    for k in k_values:
        out = np.zeros(4 * (n_steps + 1), dtype=np.float64)
        lib.manifold_kick_run_3d(
            0,                          # originSite
            sigma, k, 0.0, 0.0,        # sigma, kx, ky, kz
            mix_phi, n_steps,
            out.ctypes.data_as(c_dp))

        out = out.reshape(n_steps + 1, 4)
        mean_x, mean_y, mean_z = out[:, 0], out[:, 1], out[:, 2]

        # Fit velocity from early time
        t_fit = min(n_steps // 2, 30)
        t_arr = np.arange(t_fit)
        vx = np.polyfit(t_arr, mean_x[:t_fit], 1)[0] if t_fit > 3 else 0
        vy = np.polyfit(t_arr, mean_y[:t_fit], 1)[0] if t_fit > 3 else 0
        vz = np.polyfit(t_arr, mean_z[:t_fit], 1)[0] if t_fit > 3 else 0
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

        results[k] = (mean_x, mean_y, mean_z, vx, vy, vz)
        print(f"{k:6.2f} {vx:8.4f} {vy:8.4f} {vz:8.4f} {v_mag:8.4f}", flush=True)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for k in k_values:
        mean_x, mean_y, mean_z, vx, vy, vz = results[k]
        t = np.arange(n_steps + 1)
        axes[0].plot(t, mean_x, label=f'k={k:.1f}')
        disp = np.sqrt((mean_x - mean_x[0])**2 + (mean_y - mean_y[0])**2 + (mean_z - mean_z[0])**2)
        axes[1].plot(t, disp, label=f'k={k:.1f}')

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('⟨x⟩')
    axes[0].set_title('Wavepacket Center (x)')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('|⟨r⟩ - ⟨r⟩₀|')
    axes[1].set_title('Displacement from Start')
    axes[1].legend(fontsize=8)

    ks = sorted(results.keys())
    vs = [np.sqrt(results[k][3]**2 + results[k][4]**2 + results[k][5]**2) for k in ks]
    axes[2].plot(ks, vs, 'ko-', markersize=6)
    axes[2].set_xlabel('k (momentum)')
    axes[2].set_ylabel('|v_group|')
    axes[2].set_title('Group Velocity vs Momentum')

    fig.suptitle(f'Wavepacket Kick (3D): {tri.n_tets} tets, σ={sigma:.1f}, φ={mix_phi}', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_kick_test.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
