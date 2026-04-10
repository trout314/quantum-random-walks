#!/usr/bin/env python3
"""
Test whether the walk operator mixes A₄ irrep sectors.

Each manifold tet has 12 lattice sites (A₄ orbit). The walk state
at a tet is ψ: {12 sites} → C⁴, a C⁴⁸ object.

Coarse-graining projects onto the A₄-trivial sector (sum the 12 sites).
If the walk mixes sectors, this projection loses information.
"""
import ctypes, numpy as np, os, sys

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
    lib.manifold_zero_psi.restype = None
    lib.manifold_set_psi.argtypes = [ctypes.c_int, c_dp, c_dp]
    lib.manifold_set_psi_bulk.argtypes = [c_dp, c_dp]
    lib.manifold_get_psi.argtypes = [c_dp, c_dp]
    lib.manifold_get_site_tets.argtypes = [c_ip]
    lib.manifold_get_site_positions.argtypes = [c_dp]
    return lib


class WalkState:
    """Fast wavefunction I/O."""
    def __init__(self, lib, nsites):
        self.lib = lib
        self.nsites = nsites
        self.dim = 4 * nsites
        self._re = np.zeros(self.dim, dtype=np.float64)
        self._im = np.zeros(self.dim, dtype=np.float64)
        self._c_dp = ctypes.POINTER(ctypes.c_double)

    def set(self, psi):
        self._re[:] = psi.real
        self._im[:] = psi.imag
        self.lib.manifold_set_psi_bulk(
            self._re.ctypes.data_as(self._c_dp),
            self._im.ctypes.data_as(self._c_dp))

    def get(self):
        self.lib.manifold_get_psi(
            self._re.ctypes.data_as(self._c_dp),
            self._im.ctypes.data_as(self._c_dp))
        return self._re + 1j * self._im

    def step(self, phi):
        self.lib.manifold_step(phi)

    def evolve(self, psi, n_steps, phi):
        self.set(psi)
        for _ in range(n_steps):
            self.step(phi)
        return self.get()


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
    nchains = lib.manifold_nchains()
    dim = 4 * nsites
    print(f"Lattice: {nsites} sites, {nchains} chains, dim={dim}")
    print(f"Sites per tet: {nsites / tri.n_tets:.1f}")

    ws = WalkState(lib, nsites)

    # Site → tet mapping
    tets = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(tets.ctypes.data_as(c_ip))

    tet_sites = {}
    for s in range(nsites):
        t = tets[s]
        if t not in tet_sites:
            tet_sites[t] = []
        tet_sites[t].append(s)

    sizes = [len(v) for v in tet_sites.values()]
    print(f"All tets have 12 sites: {all(s==12 for s in sizes)}")

    # ================================================================
    # TEST A: Symmetric IC → does it stay symmetric?
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST A: A₄-symmetric IC (same spinor at all 12 sites)")
    print("=" * 60)

    chi = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)
    t0 = sorted(tet_sites.keys())[0]
    sites0 = tet_sites[t0]

    psi0 = np.zeros(dim, dtype=complex)
    for s in sites0:
        psi0[4*s:4*s+4] = chi / np.sqrt(12)
    psi0 /= np.linalg.norm(psi0)

    psi_t = psi0.copy()
    prev_steps = 0
    for n_steps in [1, 2, 5, 10, 20, 50]:
        # Evolve incrementally
        psi_t = ws.evolve(psi_t, n_steps - prev_steps, mix_phi)
        prev_steps = n_steps

        # For each tet: coefficient of variation of per-site density
        max_cv = 0
        for t in tet_sites:
            sites = tet_sites[t]
            dens = np.array([np.sum(np.abs(psi_t[4*s:4*s+4])**2) for s in sites])
            mean_d = np.mean(dens)
            if mean_d > 1e-15:
                cv = np.std(dens) / mean_d
                max_cv = max(max_cv, cv)

        status = "BROKEN" if max_cv > 0.01 else "preserved"
        print(f"  t={n_steps:3d}: max density CV = {max_cv:.6f}  [{status}]")

    # ================================================================
    # TEST B: Anti-symmetric IC → does it leak into trivial sector?
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST B: Anti-symmetric IC (zero-mean across 12 sites)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    phases = rng.randn(12)
    phases -= np.mean(phases)  # project out trivial component

    psi_anti = np.zeros(dim, dtype=complex)
    for i, s in enumerate(sites0):
        psi_anti[4*s:4*s+4] = phases[i] * chi
    psi_anti /= np.linalg.norm(psi_anti)

    def trivial_weight(psi):
        """Weight in A₄-trivial sector: Σ_tet |Σ_sites ψ(s)|² / 12."""
        w = 0
        for t in tet_sites:
            sites = tet_sites[t]
            mean_sp = np.zeros(4, dtype=complex)
            for s in sites:
                mean_sp += psi[4*s:4*s+4]
            w += np.sum(np.abs(mean_sp)**2) / 12
        return w

    print(f"  Initial trivial weight: {trivial_weight(psi_anti):.10f}")

    psi_t = psi_anti.copy()
    prev_steps = 0
    for n_steps in [1, 2, 5, 10, 20, 50]:
        psi_t = ws.evolve(psi_t, n_steps - prev_steps, mix_phi)
        prev_steps = n_steps
        tw = trivial_weight(psi_t)
        tot = np.sum(np.abs(psi_t)**2)
        print(f"  t={n_steps:3d}: trivial={tw:.10f}, total={tot:.6f}, "
              f"leak={tw:.2e}")

    # ================================================================
    # TEST C: Single-site IC → spreading among 12 images
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST C: Single-site IC (1 of 12 sites at tet 0)")
    print("=" * 60)

    s0 = sites0[0]
    psi_single = np.zeros(dim, dtype=complex)
    psi_single[4*s0:4*s0+4] = chi
    psi_single /= np.linalg.norm(psi_single)

    psi_t = psi_single.copy()
    prev_steps = 0
    for n_steps in [1, 2, 5, 10, 20, 50]:
        psi_t = ws.evolve(psi_t, n_steps - prev_steps, mix_phi)
        prev_steps = n_steps

        # At source tet
        dens0 = np.array([np.sum(np.abs(psi_t[4*s:4*s+4])**2) for s in sites0])
        n_active = np.sum(dens0 > 0.01 * np.max(dens0)) if np.max(dens0) > 1e-15 else 0
        cv0 = np.std(dens0) / np.mean(dens0) if np.mean(dens0) > 1e-15 else 0

        # Globally: fraction of total weight at source tet
        total = np.sum(np.abs(psi_t)**2)
        at_t0 = np.sum(dens0)

        print(f"  t={n_steps:3d}: source tet has {n_active:2d}/12 active, "
              f"CV={cv0:.4f}, retain={at_t0:.4f}")

    # ================================================================
    # TEST D: One step from each of the 12 sites at tet 0
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST D: One W step from each site at tet 0 separately")
    print("=" * 60)

    # For each source site, track which target sites get amplitude
    print(f"\n  Source tet {t0} has sites: {sites0}")

    all_target_dens = []
    for i, s_src in enumerate(sites0):
        psi_1 = np.zeros(dim, dtype=complex)
        psi_1[4*s_src:4*s_src+4] = np.array([1,0,0,0], dtype=complex)
        psi_out = ws.evolve(psi_1, 1, mix_phi)

        # Which tets receive amplitude?
        tet_dens = {}
        for t in tet_sites:
            sites = tet_sites[t]
            d = sum(np.sum(np.abs(psi_out[4*s:4*s+4])**2) for s in sites)
            if d > 1e-15:
                tet_dens[t] = d
        all_target_dens.append(tet_dens)

    # Check: do all 12 source sites reach the SAME set of target tets?
    target_sets = [set(d.keys()) for d in all_target_dens]
    all_same_targets = all(ts == target_sets[0] for ts in target_sets)
    print(f"  All 12 sources reach same target tets: {all_same_targets}")
    print(f"  Target tets: {sorted(target_sets[0])}")

    # For each target tet: does each source site send to exactly 1 target site,
    # or spread across multiple?
    for t_targ in sorted(target_sets[0])[:3]:
        print(f"\n  → tet {t_targ}:")
        sites_targ = tet_sites[t_targ]
        site_map = np.zeros((12, 12))  # source × target density
        for i, s_src in enumerate(sites0):
            psi_1 = np.zeros(dim, dtype=complex)
            psi_1[4*s_src:4*s_src+4] = np.array([1,0,0,0], dtype=complex)
            psi_out = ws.evolve(psi_1, 1, mix_phi)
            for j, s_targ in enumerate(sites_targ):
                site_map[i, j] = np.sum(np.abs(psi_out[4*s_targ:4*s_targ+4])**2)

        # Is it a permutation matrix (each source → exactly 1 target)?
        row_maxes = np.max(site_map, axis=1)
        row_sums = np.sum(site_map, axis=1)
        perm_quality = np.min(row_maxes / (row_sums + 1e-30))
        n_nonzero_per_row = np.sum(site_map > 1e-10, axis=1)

        print(f"    Targets per source: {n_nonzero_per_row}")
        print(f"    Permutation quality: {perm_quality:.6f} (1.0 = exact permutation)")
        print(f"    Row sums: {row_sums}")


if __name__ == '__main__':
    main()
