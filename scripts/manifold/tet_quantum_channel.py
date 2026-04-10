#!/usr/bin/env python3
"""
Compute the tet-level quantum channel from the walk operator.

The walk W is unitary on C^{4·N_sites}. We reorganize by tet:
each tet has 12 sites, so C^{48} per tet. The "observable" state
at a tet is the 4×4 reduced density matrix obtained by tracing
over the chain (site) index within the tet.

We extract Kraus operators for the tet-to-tet channel and check
trace preservation, purity evolution, and Dirac-like behavior.
"""
import ctypes, numpy as np, os, sys

from src.triangulation import Triangulation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    lib.manifold_zero_psi.restype = None
    lib.manifold_set_psi_bulk.argtypes = [c_dp, c_dp]
    lib.manifold_get_psi.argtypes = [c_dp, c_dp]
    lib.manifold_get_site_tets.argtypes = [c_ip]
    return lib


def build_walk_matrix(lib, nsites, mix_phi):
    """Build the full walk matrix column by column."""
    c_dp = ctypes.POINTER(ctypes.c_double)
    dim = 4 * nsites
    W = np.zeros((dim, dim), dtype=complex)
    re_out = np.zeros(dim, dtype=np.float64)
    im_out = np.zeros(dim, dtype=np.float64)
    re_in = np.zeros(dim, dtype=np.float64)
    im_in = np.zeros(dim, dtype=np.float64)

    for col in range(dim):
        if col % 1000 == 0:
            print(f"  col {col}/{dim}", flush=True)
        re_in[:] = 0; im_in[:] = 0
        re_in[col] = 1.0
        lib.manifold_set_psi_bulk(re_in.ctypes.data_as(c_dp), im_in.ctypes.data_as(c_dp))
        lib.manifold_step(mix_phi)
        lib.manifold_get_psi(re_out.ctypes.data_as(c_dp), im_out.ctypes.data_as(c_dp))
        W[:, col] = re_out + 1j * im_out
    return W


def tet_reduced_density(psi, tet_sites_map, tet_id):
    """Compute 4×4 reduced density matrix at a tet by tracing over chain index."""
    sites = tet_sites_map[tet_id]
    rho = np.zeros((4, 4), dtype=complex)
    for s in sites:
        sp = psi[4*s:4*s+4]
        rho += np.outer(sp, sp.conj())
    return rho


def extract_kraus_operators(W, source_tet, target_tet, tet_sites_map):
    """
    Extract Kraus operators for the channel from source_tet to target_tet.

    For each target site j at target_tet and source site i at source_tet,
    E_{j,i} = W[4j:4j+4, 4i:4i+4] is a 4×4 matrix.

    The channel acts on the FULL 48-dim state at the source tet:
    if the source state has density matrix σ (48×48), then the
    4×4 reduced density matrix at the target is:

        ρ_target = Σ_j (Σ_i E_{ji} ψ_i)(...)† = Σ_{j,i,k} E_{ji} σ_{ik} E_{jk}†

    But if we only know the 4×4 reduced density matrix at the source
    (having already traced over the chain index), the channel is:

        ρ_target = Σ_{j,i} E_{ji} ρ_source_i E_{ji}†

    where ρ_source_i = |s_i⟩⟨s_i| for each source site.

    This is NOT a function of ρ_source = Σ_i ρ_source_i alone.
    The cross terms are lost. So the tet-level channel requires
    knowing which site the amplitude is at, not just the reduced ρ.

    However, the Kraus operators E_{ji} are still useful:
    - There are 12×12 = 144 of them (4×4 each)
    - Trace preservation: Σ_{j,i} E_{ji}† E_{ji} should relate to I_{48}
    """
    source_sites = tet_sites_map[source_tet]
    target_sites = tet_sites_map[target_tet]

    kraus = []
    for j in target_sites:
        for i in source_sites:
            E = W[4*j:4*j+4, 4*i:4*i+4].copy()
            if np.linalg.norm(E) > 1e-15:
                kraus.append((j, i, E))
    return kraus


def main():
    tri_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else
        '~/Desktop/Discrete-Differential-Geometry/standard_triangulations/equilibrated_200.mfd')
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    tri = Triangulation.load(tri_path)
    print(f"Manifold: {tri.n_tets} tets, {tri.n_verts} verts")

    lib = load_lib()
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_dp = ctypes.POINTER(ctypes.c_double)

    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    nbrs_flat = np.array(tri.neighbors, dtype=np.int32).flatten()
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(c_ip), nbrs_flat.ctypes.data_as(c_ip), tri.n_tets)
    nsites = lib.manifold_build_lattice()
    dim = 4 * nsites
    n_tets = tri.n_tets
    print(f"Lattice: {nsites} sites, dim={dim}")

    # Site → tet mapping
    tets = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(tets.ctypes.data_as(c_ip))
    tet_sites = {}
    for s in range(nsites):
        t = tets[s]
        if t not in tet_sites:
            tet_sites[t] = []
        tet_sites[t].append(s)

    # Build walk matrix
    print(f"\nBuilding walk matrix ({dim}×{dim}), φ={mix_phi}...", flush=True)
    W = build_walk_matrix(lib, nsites, mix_phi)
    print(f"Unitarity: ||W†W-I|| = {np.max(np.abs(W.conj().T @ W - np.eye(dim))):.2e}")

    # ================================================================
    # 1. Kraus operators for a specific tet pair
    # ================================================================
    print("\n" + "=" * 60)
    print("1. KRAUS OPERATORS")
    print("=" * 60)

    t0 = 0
    # Find which tets receive amplitude from tet 0
    sites0 = tet_sites[t0]
    idx0 = []
    for s in sites0:
        idx0.extend(range(4*s, 4*s+4))
    idx0 = np.array(idx0)

    W_from_t0 = W[:, idx0]  # all rows, columns from tet 0
    target_tets = {}
    for t in range(n_tets):
        sites_t = tet_sites[t]
        idx_t = []
        for s in sites_t:
            idx_t.extend(range(4*s, 4*s+4))
        block_norm = np.linalg.norm(W_from_t0[idx_t])
        if block_norm > 1e-10:
            target_tets[t] = block_norm

    print(f"\nTet {t0} connects to {len(target_tets)} target tets:")
    for t, norm in sorted(target_tets.items(), key=lambda x: -x[1]):
        kraus = extract_kraus_operators(W, t0, t, tet_sites)
        n_nonzero = len(kraus)
        # Check: Σ E†E for this tet pair
        sum_EdE = np.zeros((4, 4), dtype=complex)
        for j, i, E in kraus:
            sum_EdE += E.conj().T @ E
        print(f"  → tet {t:3d}: ||block||={norm:.4f}, "
              f"{n_nonzero:3d} Kraus ops, "
              f"||Σ E†E||={np.linalg.norm(sum_EdE):.4f}")

    # Global trace preservation check:
    # Σ_{all target tets} Σ_{j,i} E_{ji}† E_{ji} should give
    # a block-diagonal matrix on the 48-dim source space
    print(f"\nGlobal trace preservation (from tet {t0}):")
    sum_all = np.zeros((48, 48), dtype=complex)
    for t in target_tets:
        for j in tet_sites[t]:
            for idx_i, i in enumerate(sites0):
                E = W[4*j:4*j+4, 4*i:4*i+4]
                for idx_k, k in enumerate(sites0):
                    F = W[4*j:4*j+4, 4*k:4*k+4]
                    sum_all[4*idx_i:4*idx_i+4, 4*idx_k:4*idx_k+4] += E.conj().T @ F
    err = np.max(np.abs(sum_all - np.eye(48)))
    print(f"  ||Σ_targets Σ_j E†_ji E_jk - I_48|| = {err:.6e}")
    print(f"  (Should be 0 if trace-preserving on full C⁴⁸)")

    # ================================================================
    # 2. Purity evolution from various initial states
    # ================================================================
    print("\n" + "=" * 60)
    print("2. PURITY EVOLUTION")
    print("=" * 60)

    chi = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

    # IC A: amplitude at all 12 sites equally (trivial sector)
    print("\nIC A: Equal amplitude at all 12 sites of tet 0")
    psi_a = np.zeros(dim, dtype=complex)
    for s in sites0:
        psi_a[4*s:4*s+4] = chi / np.sqrt(12)
    psi_a /= np.linalg.norm(psi_a)

    # IC B: amplitude at just 1 site
    print("IC B: Amplitude at site 0 only")
    psi_b = np.zeros(dim, dtype=complex)
    psi_b[4*sites0[0]:4*sites0[0]+4] = chi

    # IC C: random superposition across 12 sites
    print("IC C: Random superposition across 12 sites")
    rng = np.random.RandomState(42)
    psi_c = np.zeros(dim, dtype=complex)
    for s in sites0:
        rand_sp = rng.randn(4) + 1j * rng.randn(4)
        psi_c[4*s:4*s+4] = rand_sp
    psi_c /= np.linalg.norm(psi_c)

    n_steps = 100
    ics = [('A: uniform', psi_a), ('B: single-site', psi_b), ('C: random', psi_c)]
    purity_data = {}
    trace_data = {}

    for label, psi0 in ics:
        purities = []
        traces = []
        psi = psi0.copy()
        for t_step in range(n_steps + 1):
            # Compute reduced density matrices at all tets
            total_trace = 0
            total_purity_weighted = 0
            max_trace_tet = -1
            max_trace_val = 0
            for t in range(n_tets):
                rho = tet_reduced_density(psi, tet_sites, t)
                tr = np.real(np.trace(rho))
                pur = np.real(np.trace(rho @ rho))
                total_trace += tr
                if tr > max_trace_val:
                    max_trace_val = tr
                    max_trace_tet = t
                    max_purity = pur / max(tr**2, 1e-30)
            purities.append(max_purity)
            traces.append(total_trace)

            if t_step < n_steps:
                psi = W @ psi

        purity_data[label] = purities
        trace_data[label] = traces
        print(f"\n  {label}:")
        print(f"    t=0:   purity={purities[0]:.6f}, total Tr={traces[0]:.6f}")
        print(f"    t=10:  purity={purities[10]:.6f}, total Tr={traces[10]:.6f}")
        print(f"    t=50:  purity={purities[50]:.6f}, total Tr={traces[50]:.6f}")
        print(f"    t=100: purity={purities[100]:.6f}, total Tr={traces[100]:.6f}")

    # ================================================================
    # 3. Detailed density matrix at peak tet over time
    # ================================================================
    print("\n" + "=" * 60)
    print("3. DENSITY MATRIX AT SOURCE TET")
    print("=" * 60)

    # Use IC A, track rho at tet 0
    psi = psi_a.copy()
    print(f"\n  t | Tr(ρ)    | Purity   | rank(ρ>0.01) | eigenvalues")
    for t_step in range(21):
        rho = tet_reduced_density(psi, tet_sites, t0)
        tr = np.real(np.trace(rho))
        pur = np.real(np.trace(rho @ rho))
        evals = np.sort(np.real(np.linalg.eigvalsh(rho)))[::-1]
        rank = np.sum(evals > 0.01 * evals[0]) if evals[0] > 1e-15 else 0
        if t_step <= 5 or t_step % 5 == 0:
            ev_str = ', '.join(f'{e:.4f}' for e in evals if e > 1e-6)
            print(f"  {t_step:2d} | {tr:.6f} | {pur:.6f} | {rank:12d} | {ev_str}")
        psi = W @ psi

    # ================================================================
    # Plot
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for label, purs in purity_data.items():
        ax.plot(purs[:50], label=label, lw=1.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Purity Tr(ρ²)/Tr(ρ)²')
    ax.set_title('Purity of densest tet')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15)
    ax.axhline(0.25, color='gray', ls='--', alpha=0.3, label='maximally mixed (4D)')

    ax = axes[1]
    for label, trs in trace_data.items():
        ax.plot(trs[:50], label=label, lw=1.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Σ_tets Tr(ρ_tet)')
    ax.set_title('Total trace (= full norm²)')
    ax.legend()
    ax.grid(True, alpha=0.15)

    plt.suptitle('Tet-level density matrix: quantum channel analysis', fontsize=14)
    plt.tight_layout()
    out = '/tmp/tet_quantum_channel.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
