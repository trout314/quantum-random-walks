#!/usr/bin/env python3
"""
Wavepacket kick test on the large 100k-tet manifold.
Adapted from manifold_kick_test.py with parameters scaled for 1.2M sites.
"""
import ctypes, numpy as np, os, sys

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
    lib.manifold_get_site_positions.argtypes = [c_dp]
    lib.manifold_kick_run_3d.argtypes = [
        ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_int, c_dp]
    lib.manifold_mean_position.argtypes = [c_dp]
    return lib


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

    # Get site positions to check spatial scale
    pos = np.zeros(3 * nsites, dtype=np.float64)
    lib.manifold_get_site_positions(pos.ctypes.data_as(c_dp))
    pos = pos.reshape(nsites, 3)

    r_from_origin = np.sqrt((pos**2).sum(axis=1))
    print(f"Position range: x=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}]"
          f"  y=[{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]"
          f"  z=[{pos[:,2].min():.1f}, {pos[:,2].max():.1f}]")
    print(f"Max |r| from site 0: {r_from_origin.max():.1f}")

    # Parameters scaled for 100k manifold
    # m ≈ 0.878 * phi = 0.044, Compton wavelength 1/m ≈ 23
    mix_phi = 0.05
    n_steps = 100
    sigma = 15.0   # several Compton wavelengths wide — smooth wavepacket

    # Kick along x with k comparable to and exceeding m
    k_values = [0.0, 0.02, 0.05, 0.10, 0.20]

    print(f"\nσ={sigma:.1f}, φ_mix={mix_phi}, {n_steps} steps")
    print(f"m ≈ {0.878*mix_phi:.4f}, 1/m ≈ {1/(0.878*mix_phi):.1f}")
    print(f"Origin site: 0 at ({pos[0,0]:.2f}, {pos[0,1]:.2f}, {pos[0,2]:.2f})")
    print(f"{'k':>6} {'v_x':>8} {'v_y':>8} {'v_z':>8} {'|v|':>8} {'norm_f':>10}")

    results = {}
    for k in k_values:
        out = np.zeros(4 * (n_steps + 1), dtype=np.float64)
        lib.manifold_kick_run_3d(
            0, sigma, k, 0.0, 0.0,
            mix_phi, n_steps,
            out.ctypes.data_as(c_dp))

        out = out.reshape(n_steps + 1, 4)
        mean_x, mean_y, mean_z, mean_r2 = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        # Check norm after run
        norm_f = np.sqrt(lib.manifold_norm2())

        # Fit velocity from linear regime (skip first few steps for transient)
        t_fit = min(n_steps, 80)
        t_arr = np.arange(5, t_fit)
        vx = np.polyfit(t_arr, mean_x[5:t_fit], 1)[0]
        vy = np.polyfit(t_arr, mean_y[5:t_fit], 1)[0]
        vz = np.polyfit(t_arr, mean_z[5:t_fit], 1)[0]
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

        results[k] = (mean_x, mean_y, mean_z, mean_r2, vx, vy, vz)
        print(f"{k:6.3f} {vx:8.4f} {vy:8.4f} {vz:8.4f} {v_mag:8.4f} {norm_f:10.7f}",
              flush=True)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    t = np.arange(n_steps + 1)
    base_x, base_y, base_z = results[0.0][0], results[0.0][1], results[0.0][2]

    # Raw <x>(t) for each k
    ax = axes[0, 0]
    for k in k_values:
        ax.plot(t, results[k][0], label=f'k={k:.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('⟨x⟩')
    ax.set_title('Mean x-position')
    ax.legend(fontsize=8)

    # Baseline-subtracted displacement along kick direction
    ax = axes[0, 1]
    for k in k_values[1:]:
        dx = results[k][0] - base_x
        ax.plot(t, dx, label=f'k={k:.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('⟨x⟩ − ⟨x⟩_{k=0}')
    ax.set_title('Kick effect (baseline subtracted)')
    ax.legend(fontsize=8)

    # Spread: <r²> - <r>² proxy
    ax = axes[1, 0]
    for k in k_values:
        r2 = results[k][3]
        rx = results[k][0]
        ry = results[k][1]
        rz = results[k][2]
        spread = r2 - rx**2 - ry**2 - rz**2
        ax.plot(t, spread, label=f'k={k:.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('⟨r²⟩ − ⟨r⟩²')
    ax.set_title('Wavepacket spread')
    ax.legend(fontsize=8)

    # Group velocity vs k
    ax = axes[1, 1]
    ks = sorted(results.keys())
    vxs = [results[k][4] for k in ks]
    v_mags = [np.sqrt(results[k][4]**2 + results[k][5]**2 + results[k][6]**2) for k in ks]
    ax.plot(ks, vxs, 'bo-', markersize=6, label='v_x')
    ax.plot(ks, v_mags, 'rs-', markersize=6, label='|v|')
    # Dirac prediction: v = k/sqrt(k² + m²)
    m = 0.878 * mix_phi
    k_theory = np.linspace(0, max(ks) * 1.1, 100)
    v_theory = k_theory / np.sqrt(k_theory**2 + m**2)
    ax.plot(k_theory, v_theory, 'k--', alpha=0.5, label=f'k/√(k²+m²), m={m:.3f}')
    ax.set_xlabel('k (momentum)')
    ax.set_ylabel('Group velocity')
    ax.set_title('Dispersion relation')
    ax.legend(fontsize=8)

    fig.suptitle(f'Wavepacket Kick: {tri.n_tets} tets, {nsites} sites, '
                 f'σ={sigma:.0f}, φ={mix_phi}', fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_kick_100k.png'
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
