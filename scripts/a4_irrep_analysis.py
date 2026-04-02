#!/usr/bin/env python3
"""
A₄ irrep decomposition of the walk operator on a manifold.

For each manifold tetrahedron, 12 lattice sites form an A₄ orbit.
The walk state at a tet is ψ: A₄ → C⁴, living in C⁴⁸.

The regular representation of A₄ decomposes as:
    C¹² = 1 ⊕ 1' ⊕ 1'' ⊕ 3 ⊕ 3 ⊕ 3

Question: does the walk operator mix these irrep sectors?
If yes, the full C⁴⁸ per tet is fundamental.
If no, the trivial-sector C⁴ suffices (coarse-graining is exact).
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
    lib.manifold_get_site_tets.argtypes = [c_ip]
    lib.manifold_get_site_positions.argtypes = [c_dp]
    return lib


def build_walk_matrix(lib, nsites, mix_phi):
    """Build the full walk matrix by applying W to each basis vector."""
    dim = 4 * nsites
    W = np.zeros((dim, dim), dtype=complex)
    c_dp = ctypes.POINTER(ctypes.c_double)
    out_re = np.zeros(dim, dtype=np.float64)
    out_im = np.zeros(dim, dtype=np.float64)
    re4 = np.zeros(4, dtype=np.float64)
    im4 = np.zeros(4, dtype=np.float64)

    for col in range(dim):
        site = col // 4
        comp = col % 4
        lib.manifold_zero_psi()
        re4[:] = 0; re4[comp] = 1.0; im4[:] = 0
        lib.manifold_set_psi(site,
                             re4.ctypes.data_as(c_dp),
                             im4.ctypes.data_as(c_dp))
        lib.manifold_step(mix_phi)
        lib.manifold_get_psi(out_re.ctypes.data_as(c_dp),
                             out_im.ctypes.data_as(c_dp))
        W[:, col] = out_re + 1j * out_im
    return W


def get_site_tets(lib, nsites):
    """Get site → tet_id mapping."""
    c_ip = ctypes.POINTER(ctypes.c_int)
    tets = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(tets.ctypes.data_as(c_ip))
    return tets


def get_site_positions(lib, nsites):
    """Get site positions."""
    c_dp = ctypes.POINTER(ctypes.c_double)
    pos = np.zeros(3 * nsites, dtype=np.float64)
    lib.manifold_get_site_positions(pos.ctypes.data_as(c_dp))
    return pos.reshape(nsites, 3)


# Standard tetrahedral directions (unit vectors from centroid to vertices)
def tet_directions():
    """Standard tetrahedral vertex directions."""
    d0 = np.array([0, 0, 1.0])
    d1 = np.array([2*np.sqrt(2)/3, 0, -1/3])
    d2 = np.array([-np.sqrt(2)/3, np.sqrt(6)/3, -1/3])
    d3 = np.array([-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3])
    return np.array([d0, d1, d2, d3])


# All 12 even permutations of {0,1,2,3}
A4_PERMS = [
    [0,1,2,3], [1,2,0,3], [2,0,1,3],
    [1,3,2,0], [3,0,2,1], [2,1,3,0],
    [3,1,0,2], [0,2,3,1], [0,3,1,2],
    [1,0,3,2], [2,3,0,1], [3,2,1,0],
]

def a4_rotation_matrix(perm):
    """3×3 rotation matrix for an A₄ permutation of tet vertex directions."""
    dirs = tet_directions()
    # R maps dirs[k] → dirs[perm[k]]
    # R = (3/4) Σ_k dirs[perm[k]] ⊗ dirs[k]
    R = np.zeros((3, 3))
    for k in range(4):
        R += np.outer(dirs[perm[k]], dirs[k])
    R *= 3.0 / 4.0
    return R


def a4_spinor_rotation(R):
    """
    Compute the SU(2) spinor rotation corresponding to a 3×3 rotation R.

    For the 4×4 Dirac spinor with τ = ν β + (3/4)(d·α),
    the A₄ rotation acts as a block-diagonal SU(2)×SU(2) matrix.

    Actually, for the walk's spinor representation, we need the full 4×4
    transformation that maps τ(d) → τ(Rd). This is the representation
    of SO(3) on the spinor space.
    """
    # For Dirac spinors: the rotation acts via exp(-i θ/2 n̂·Σ)
    # where Σ = (1/2) σ ⊗ I₂ (spin generators in 4×4)
    # But our α matrices define the representation.

    # Extract rotation axis and angle from R
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_matrix(R)
    rotvec = r.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(4, dtype=complex)

    axis = rotvec / angle

    # Spin-1/2 generators in 4×4 block diagonal (SU(2) × SU(2))
    # Using the standard Dirac representation where
    # Σ_i = (1/2) [[σ_i, 0], [0, σ_i]]
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]

    Sigma = np.zeros((4, 4), dtype=complex)
    for i in range(3):
        block = 0.5 * axis[i] * sigma[i]
        Sigma[:2, :2] += block
        Sigma[2:, 2:] += block

    U = np.eye(4, dtype=complex) * np.cos(angle/2) - 1j * np.sin(angle/2) * 2 * Sigma
    # Actually: exp(-i angle/2 n̂·Σ) = cos(θ/2) I - i sin(θ/2) n̂·(2Σ)
    # But 2Σ_i = [[σ_i, 0], [0, σ_i]], so:
    s = 0.5 * (axis[0] * sigma[0] + axis[1] * sigma[1] + axis[2] * sigma[2])
    U2 = np.zeros((4, 4), dtype=complex)
    block = np.cos(angle/2) * np.eye(2) - 1j * np.sin(angle/2) * (axis[0]*sigma[0] + axis[1]*sigma[1] + axis[2]*sigma[2])
    U2[:2, :2] = block
    U2[2:, 2:] = block
    return U2


def main():
    tri_path = os.path.expanduser(
        sys.argv[1] if len(sys.argv) > 1 else
        '~/Desktop/Discrete-Differential-Geometry/standard_triangulations/dim_3_sphere.mfd')
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
    print(f"Lattice: {nsites} sites, {nchains} chains")
    print(f"Sites per tet: {nsites / tri.n_tets:.1f}")

    dim = 4 * nsites
    print(f"\nBuilding walk matrix ({dim}×{dim}), φ={mix_phi}...")
    W = build_walk_matrix(lib, nsites, mix_phi)

    # Unitarity check
    WdW = W.conj().T @ W
    print(f"Unitarity: ||W†W - I|| = {np.max(np.abs(WdW - np.eye(dim))):.2e}")

    # Get site → tet mapping
    site_tets = get_site_tets(lib, nsites)
    positions = get_site_positions(lib, nsites)

    # Group sites by tet
    tet_sites = {}
    for s in range(nsites):
        t = site_tets[s]
        if t not in tet_sites:
            tet_sites[t] = []
        tet_sites[t].append(s)

    print(f"\nSites per tet:")
    for t in sorted(tet_sites.keys()):
        print(f"  tet {t}: {len(tet_sites[t])} sites")

    # ================================================================
    # Test 1: Does W map tet-grouped states to tet-grouped states?
    # i.e., for each tet, what fraction of the output lands on
    # the same tet vs other tets?
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Walk operator locality (tet-to-tet coupling)")
    print("=" * 60)

    for t in sorted(tet_sites.keys())[:3]:
        sites = tet_sites[t]
        # Indices in the 4N-dim Hilbert space for this tet
        idx = []
        for s in sites:
            idx.extend([4*s, 4*s+1, 4*s+2, 4*s+3])
        idx = np.array(idx)

        # What fraction of W's output from this tet stays in this tet?
        W_from_t = W[:, idx]  # columns for this tet
        norm_total = np.linalg.norm(W_from_t)**2

        # For each target tet
        for t2 in sorted(tet_sites.keys()):
            sites2 = tet_sites[t2]
            idx2 = []
            for s in sites2:
                idx2.extend([4*s, 4*s+1, 4*s+2, 4*s+3])
            idx2 = np.array(idx2)
            norm_t2 = np.linalg.norm(W_from_t[idx2])**2
            frac = norm_t2 / norm_total
            if frac > 0.01:
                print(f"  tet {t} → tet {t2}: {frac:.4f}")

    # ================================================================
    # Test 2: A₄ equivariance test
    # If W commutes with A₄, then W maps the trivial irrep to itself.
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: A₄ equivariance (does trivial sector decouple?)")
    print("=" * 60)

    # Strategy: construct an A₄-symmetric initial state at one tet,
    # evolve with W, and check if the result is still A₄-symmetric
    # at each tet.
    #
    # A₄-symmetric means: all 12 sites at a tet carry the SAME spinor
    # (in the frame-transported sense). For simplicity, we test with
    # all 12 sites having the same raw spinor — this is an approximation
    # but will show if there's any mixing.

    # Actually, let's do this more carefully using the walk matrix directly.
    # For each tet, construct the "coarse-grained projector":
    # P_CG sums the 12 sites at each tet into one representative.

    n_tets = tri.n_tets
    # Coarse-graining matrix: C is (4*n_tets) × (4*nsites)
    # C maps the full state to the tet-level state by summing sites per tet
    C_mat = np.zeros((4 * n_tets, 4 * nsites), dtype=complex)
    for t in range(n_tets):
        sites = tet_sites.get(t, [])
        for s in sites:
            for a in range(4):
                C_mat[4*t + a, 4*s + a] += 1.0 / np.sqrt(len(sites))

    # The "embedding" is C†: takes a tet-level state and distributes
    # it equally among the 12 sites
    # If W commutes with A₄, then C W C† should be a well-defined
    # "coarse-grained walk operator" and C W (I - C†C) should be zero.

    # C†C is the projector onto the "uniform per tet" subspace
    P_sym = C_mat.conj().T @ C_mat  # projects onto A₄-trivial sector

    # Check: does W preserve the symmetric subspace?
    # If P_sym W P_sym = W P_sym (mod the complement), then yes.
    W_sym = P_sym @ W @ P_sym  # walk restricted to symmetric sector
    W_leak = (np.eye(dim) - P_sym) @ W @ P_sym  # leakage out of symmetric sector

    leak_norm = np.linalg.norm(W_leak)
    sym_norm = np.linalg.norm(W_sym)
    print(f"\n||W restricted to trivial sector|| = {sym_norm:.6f}")
    print(f"||Leakage from trivial sector||    = {leak_norm:.6f}")
    print(f"Leakage fraction: {leak_norm / (sym_norm + 1e-15):.6f}")

    if leak_norm > 1e-10:
        print("\n*** A₄ SECTORS MIX: the walk operator does NOT preserve")
        print("*** the trivial irrep. The full C⁴⁸ per tet is fundamental.")
    else:
        print("\n    A₄ sectors decouple: coarse-graining is exact.")

    # ================================================================
    # Test 3: Coarse-grained walk operator
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Coarse-grained walk operator W_CG = C W C†")
    print("=" * 60)

    W_CG = C_mat @ W @ C_mat.conj().T
    print(f"W_CG shape: {W_CG.shape}")
    WcgdWcg = W_CG.conj().T @ W_CG
    print(f"||W_CG†W_CG - I|| = {np.max(np.abs(WcgdWcg - np.eye(4*n_tets))):.6f}")
    print("(If >0, coarse-grained walk is NOT unitary → sectors must mix)")

    evals_cg = np.linalg.eigvals(W_CG)
    mags_cg = np.abs(evals_cg)
    print(f"|λ_CG| range: [{mags_cg.min():.6f}, {mags_cg.max():.6f}]")

    # ================================================================
    # Test 4: Per-tet block structure
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Intra-tet vs inter-tet coupling strength")
    print("=" * 60)

    # For each pair of tets, compute ||W_{t1,t2}|| (the block of W
    # connecting sites at t1 to sites at t2)
    for t1 in sorted(tet_sites.keys())[:2]:
        sites1 = tet_sites[t1]
        idx1 = []
        for s in sites1:
            idx1.extend(range(4*s, 4*s+4))
        idx1 = np.array(idx1)

        print(f"\n  From tet {t1} ({len(sites1)} sites):")
        for t2 in sorted(tet_sites.keys()):
            sites2 = tet_sites[t2]
            idx2 = []
            for s in sites2:
                idx2.extend(range(4*s, 4*s+4))
            idx2 = np.array(idx2)

            block = W[np.ix_(idx2, idx1)]
            fnorm = np.linalg.norm(block)
            if fnorm > 1e-10:
                print(f"    → tet {t2}: ||W_block||_F = {fnorm:.6f}")

    # ================================================================
    # Test 5: Symmetry-breaking measurement
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 5: A₄ symmetry of evolved states")
    print("=" * 60)

    # Start with uniform state at one tet, evolve, measure variance
    # within each tet (should be 0 if A₄ preserved)
    t0 = sorted(tet_sites.keys())[0]
    sites0 = tet_sites[t0]
    chi = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

    psi = np.zeros(dim, dtype=complex)
    for s in sites0:
        psi[4*s:4*s+4] = chi / np.sqrt(len(sites0))
    psi /= np.linalg.norm(psi)

    for t_step in [1, 2, 5, 10]:
        psi_t = psi.copy()
        for _ in range(t_step):
            psi_t = W @ psi_t

        # Measure density variance within each tet
        max_var = 0
        for t in sorted(tet_sites.keys()):
            sites = tet_sites[t]
            densities = []
            for s in sites:
                d = np.sum(np.abs(psi_t[4*s:4*s+4])**2)
                densities.append(d)
            densities = np.array(densities)
            if np.mean(densities) > 1e-15:
                cv = np.std(densities) / np.mean(densities)  # coeff of variation
                max_var = max(max_var, cv)

        print(f"  t={t_step:2d}: max coefficient of variation within a tet = {max_var:.6f}")
        if max_var > 0.01:
            print(f"         → A₄ symmetry BROKEN (sites within a tet have different densities)")

    out = '/tmp/a4_irrep_analysis.txt'
    print(f"\nAnalysis complete.")


if __name__ == '__main__':
    main()
