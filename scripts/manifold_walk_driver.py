#!/usr/bin/env python3
"""
Manifold walk driver: build closed chains on a triangulated 3-manifold,
run the walk in D, and coarse-grain the result in Python.
"""
import numpy as np
import ctypes, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from triangulation import Triangulation
from manifold_walk import make_tau, frame_transport, STD_DIRS, L_PERM


# ---- Chain tracing through the manifold ----

def trace_chain(tri, start_tet, start_perm):
    """Trace a chain through the manifold until it loops.

    At each step: drop vertex at window position 0 (cross opposite face),
    shift window, pick up new vertex from neighbor tet.

    Returns list of (tet_id, perm_tuple) states.  The chain loops when it
    returns to (start_tet, start_perm).
    """
    chain = []
    tet_id = start_tet
    perm = list(start_perm)

    while True:
        state = (tet_id, tuple(perm))
        if len(chain) > 0 and state == chain[0]:
            break
        chain.append(state)

        # Drop vertex at window position 0
        dropped = perm[0]
        tet_verts = list(tri.tets[tet_id])
        face_idx = tet_verts.index(dropped)
        neighbor_id = tri.neighbors[tet_id][face_idx]
        neighbor_verts = list(tri.tets[neighbor_id])
        shared = set(tet_verts) - {dropped}
        new_vert = [v for v in neighbor_verts if v not in shared][0]

        # Shift window: drop position 0, append new vertex
        perm = [perm[1], perm[2], perm[3], new_vert]
        tet_id = neighbor_id

    return chain


def exit_direction_for_state(tri, tet_id, perm):
    """Compute the exit direction for a site at (tet_id, perm).

    The exit direction points toward vertex perm[0] (the one being dropped).
    In the standard tetrahedral frame, this is STD_DIRS[k] where k is the
    position of perm[0] in the sorted tet vertex list.
    """
    tet_verts = list(tri.tets[tet_id])
    k = tet_verts.index(perm[0])
    return STD_DIRS[k]


def l_perm_of(r_perm):
    """Apply L_PERM = [1,3,0,2] to an R-chain vertex permutation."""
    return tuple(r_perm[k] for k in L_PERM)


# ---- Build the full lattice ----

def build_manifold_lattice(tri, start_tet=0):
    """Build all closed R and L chains covering the manifold.

    Each site is identified by its R-chain state (tet_id, r_perm).
    The L-chain passing through the same site has state
    (tet_id, l_perm_of(r_perm)).  Both chains share one site ID.

    Returns:
        sites: dict (tet_id, r_perm_tuple) → site_id
        r_chains: list of lists of site_ids
        l_chains: list of lists of site_ids
        exit_dirs_r: dict site_id → ndarray(3,)
        exit_dirs_l: dict site_id → ndarray(3,)
    """
    start_perm = tuple(tri.tets[start_tet])
    sites = {}          # (tet_id, r_perm) → site_id
    l_to_r = {}         # (tet_id, l_perm) → (tet_id, r_perm)
    r_chains = []       # each is a list of site_ids
    l_chains = []       # each is a list of site_ids
    exit_dirs_r = {}
    exit_dirs_l = {}
    next_site_id = 0
    inv_l = [2, 0, 3, 1]  # inverse of L_PERM

    visited_r = set()   # R-chain states already on a chain
    visited_l = set()   # L-chain states already on a chain

    def get_or_create_site(tet_id, r_perm):
        """Get or create a site from its R-chain state."""
        nonlocal next_site_id
        key = (tet_id, tuple(r_perm))
        if key not in sites:
            sid = next_site_id
            next_site_id += 1
            sites[key] = sid
            # Also register the L-state → R-state mapping
            l_perm = l_perm_of(r_perm)
            l_to_r[(tet_id, l_perm)] = key
        return sites[key]

    def r_state_of_l_state(l_state):
        """Convert an L-chain state to the corresponding R-chain state."""
        tet_id, l_perm = l_state
        r_perm = tuple(l_perm[k] for k in inv_l)
        return (tet_id, r_perm)

    r_chain_states_list = []  # parallel to r_chains, stores (tet, perm) states

    def process_r_chain(start_t, start_p):
        """Trace an R-chain if not already visited. Returns True if new."""
        if (start_t, start_p) in visited_r:
            return False
        r_chain_states = trace_chain(tri, start_t, start_p)
        for state in r_chain_states:
            visited_r.add(state)
        chain_sids = []
        for state in r_chain_states:
            sid = get_or_create_site(state[0], state[1])
            chain_sids.append(sid)
            exit_dirs_r[sid] = exit_direction_for_state(tri, *state)
        r_chains.append(chain_sids)
        r_chain_states_list.append(r_chain_states)
        return True

    def process_l_chain(start_t, start_p):
        """Trace an L-chain if not already visited. Returns new R-states."""
        if (start_t, start_p) in visited_l:
            return []
        l_chain_states = trace_chain(tri, start_t, start_p)
        for state in l_chain_states:
            visited_l.add(state)
        chain_sids = []
        new_r_states = []
        for l_state in l_chain_states:
            r_state = r_state_of_l_state(l_state)
            sid = get_or_create_site(r_state[0], r_state[1])
            chain_sids.append(sid)
            exit_dirs_l[sid] = exit_direction_for_state(tri, *l_state)
            if r_state not in visited_r:
                new_r_states.append(r_state)
        l_chains.append(chain_sids)
        return new_r_states

    # BFS: seed with first R-chain, discover L-chains, which may
    # reveal new R-chain starting states
    r_queue = [(start_tet, start_perm)]
    while r_queue:
        t, p = r_queue.pop(0)
        if not process_r_chain(t, p):
            continue
        print(f"  R-chain {len(r_chains)-1}: period={len(r_chains[-1])}, "
              f"sites={len(sites)}", flush=True)
        # Trace L-chains at each R-chain site
        for r_state in r_chain_states_list[-1]:
            l_start = l_perm_of(r_state[1])
            new_r = process_l_chain(r_state[0], l_start)
            for rs in new_r:
                if rs not in visited_r:
                    r_queue.append(rs)

    return sites, r_chains, l_chains, exit_dirs_r, exit_dirs_l


# ---- D library interface ----

def load_d_library():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, '..', 'dlang', 'build', 'quantum_walk.so')
    lib = ctypes.CDLL(lib_path)

    lib.manifold_create.argtypes = [ctypes.c_int]
    lib.manifold_create.restype = None

    lib.manifold_nsites.argtypes = []
    lib.manifold_nsites.restype = ctypes.c_int

    lib.manifold_nchains.argtypes = []
    lib.manifold_nchains.restype = ctypes.c_int

    lib.manifold_alloc_site.argtypes = [ctypes.c_double]*3
    lib.manifold_alloc_site.restype = ctypes.c_int

    lib.manifold_add_closed_chain.argtypes = [
        ctypes.POINTER(ctypes.c_int),    # siteIds
        ctypes.POINTER(ctypes.c_double), # exitDirs
        ctypes.c_int,                    # nSites
        ctypes.c_bool,                   # isR
    ]
    lib.manifold_add_closed_chain.restype = ctypes.c_int

    lib.manifold_set_psi.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.manifold_set_psi.restype = None

    lib.manifold_get_psi.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.manifold_get_psi.restype = None

    lib.manifold_norm2.argtypes = []
    lib.manifold_norm2.restype = ctypes.c_double

    lib.manifold_step.argtypes = [ctypes.c_double]
    lib.manifold_step.restype = ctypes.c_double

    lib.manifold_run_observe.argtypes = [
        ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.manifold_run_observe.restype = None

    return lib


def send_lattice_to_d(lib, sites, r_chains, l_chains, exit_dirs_r, exit_dirs_l):
    """Allocate sites and create closed chains in D."""
    nsites = len(sites)
    lib.manifold_create(nsites + 1000)  # small headroom

    # Allocate sites (in order of site ID)
    state_by_id = {v: k for k, v in sites.items()}
    for sid in range(nsites):
        # Position: use (0,0,0) — we don't need spatial positions for
        # the manifold walk (the topology is in the chain structure)
        lib.manifold_alloc_site(0.0, 0.0, 0.0)

    # Create R-chains (r_chains are already lists of site IDs)
    for chain_sids in r_chains:
        n = len(chain_sids)
        ids = (ctypes.c_int * n)(*chain_sids)
        dirs = (ctypes.c_double * (3*n))()
        for i, sid in enumerate(chain_sids):
            d = exit_dirs_r[sid]
            dirs[3*i], dirs[3*i+1], dirs[3*i+2] = d[0], d[1], d[2]
        lib.manifold_add_closed_chain(ids, dirs, n, True)

    # Create L-chains
    for chain_sids in l_chains:
        n = len(chain_sids)
        ids = (ctypes.c_int * n)(*chain_sids)
        dirs = (ctypes.c_double * (3*n))()
        for i, sid in enumerate(chain_sids):
            d = exit_dirs_l[sid]
            dirs[3*i], dirs[3*i+1], dirs[3*i+2] = d[0], d[1], d[2]
        lib.manifold_add_closed_chain(ids, dirs, n, False)

    print(f"  D lattice: {lib.manifold_nsites()} sites, {lib.manifold_nchains()} chains")


# ---- Coarse-graining ----

def coarse_grain(sites, psi_re, psi_im, exit_dirs_r):
    """Sum spinor amplitudes at each manifold tet with frame transport.

    Returns dict: tet_id → complex spinor (4,).
    """
    # Group sites by manifold tet
    tet_sites = {}  # tet_id → list of (site_id, exit_dir, spinor)
    state_by_id = {v: k for k, v in sites.items()}
    for sid in range(len(psi_re) // 4):
        state = state_by_id.get(sid)
        if state is None:
            continue
        tet_id = state[0]
        spinor = psi_re[4*sid:4*sid+4] + 1j * psi_im[4*sid:4*sid+4]
        if np.linalg.norm(spinor) < 1e-30:
            continue
        d = exit_dirs_r.get(sid)
        if d is None:
            continue
        if tet_id not in tet_sites:
            tet_sites[tet_id] = []
        tet_sites[tet_id].append((d, spinor))

    # For each tet, pick first site's frame as reference, transport others
    manifold_psi = {}
    for tet_id, site_list in tet_sites.items():
        ref_dir = site_list[0][0]
        tau_ref = make_tau(ref_dir)
        total = site_list[0][1].copy()

        for d, spinor in site_list[1:]:
            tau_s = make_tau(d)
            U = frame_transport(tau_s, tau_ref)
            total += U @ spinor

        manifold_psi[tet_id] = total

    return manifold_psi


# ---- Main ----

def main():
    default_tri = ('~/Desktop/Discrete-Differential-Geometry/'
                   'standard_triangulations/equilibrated_200.mfd')
    tri_path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else default_tri)
    tri = Triangulation.load(tri_path)
    print(f"Manifold: {tri.n_tets} tets, {tri.n_verts} verts")

    # Load D library
    lib = load_d_library()

    # Pass triangulation to D for fast chain tracing
    lib.manifold_load_triangulation.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.manifold_load_triangulation.restype = None

    lib.manifold_build_lattice.argtypes = []
    lib.manifold_build_lattice.restype = ctypes.c_int

    # Pack triangulation data as flat int arrays
    tets_flat = np.array(tri.tets, dtype=np.int32).flatten()
    neighbors_flat = np.array(tri.neighbors, dtype=np.int32).flatten()

    print("Loading triangulation into D...")
    lib.manifold_load_triangulation(
        tets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        neighbors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tri.n_tets)

    print("Building manifold lattice (chain tracing in D)...")
    nsites = lib.manifold_build_lattice()
    nchains = lib.manifold_nchains()
    print(f"  {nsites} sites ({nsites/tri.n_tets:.0f}× tets), {nchains} chains")

    # Initial condition: balanced spinor (1, 0, i, 0)/√2 at site 0
    ic_re = np.array([1/np.sqrt(2), 0, 0, 0], dtype=np.float64)
    ic_im = np.array([0, 0, 1/np.sqrt(2), 0], dtype=np.float64)
    lib.manifold_set_psi(0,
                         ic_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         ic_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # Walk parameters
    n_steps = 200
    mix_phi = 0.05

    print(f"\nWalk: {n_steps} steps, φ_mix={mix_phi}, {nsites} sites")

    # Run all steps in D with observables
    norms = np.zeros(n_steps, dtype=np.float64)
    pr = np.zeros(n_steps, dtype=np.float64)
    ret_prob = np.zeros(n_steps, dtype=np.float64)

    print(f"  t=0 norm={np.sqrt(lib.manifold_norm2()):12.9f}", flush=True)
    lib.manifold_run_observe(
        n_steps, mix_phi,
        norms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ret_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # Print selected steps
    t_arr = np.arange(1, n_steps + 1)
    print(f"{'t':>5} {'norm':>12} {'PR':>10} {'PR/N':>8} {'return':>12}")
    for t in range(n_steps):
        if t < 5 or t % 50 == 0 or t == n_steps - 1:
            print(f"{t+1:5d} {np.sqrt(norms[t]):12.9f} {pr[t]:10.1f} "
                  f"{pr[t]/nsites:8.4f} {ret_prob[t]:12.6e}")

    print(f"\n  Final norm: {np.sqrt(norms[-1]):.15f}")
    print(f"  Max |norm-1|: {np.max(np.abs(norms - 1.0)):.2e}")
    print(f"  PR range: {pr.min():.1f} — {pr.max():.1f} (N={nsites}, N/3={nsites/3:.0f})")

    # Get final wavefunction for coarse-grained histogram
    lib.manifold_site_probs.argtypes = [ctypes.POINTER(ctypes.c_double)]
    lib.manifold_site_probs.restype = ctypes.c_double
    site_probs = np.zeros(nsites, dtype=np.float64)
    lib.manifold_site_probs(site_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # Get site→tet mapping from D
    lib.manifold_get_site_tets.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.manifold_get_site_tets.restype = None
    site_tets = np.zeros(nsites, dtype=np.int32)
    lib.manifold_get_site_tets(site_tets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

    # Coarse-grain: sum site probabilities at each tet
    tet_probs = np.zeros(tri.n_tets, dtype=np.float64)
    for sid in range(nsites):
        tet_probs[site_tets[sid]] += site_probs[sid]

    # Compute vertex degrees for correlation
    from collections import Counter
    vert_deg = Counter()
    for t in tri.tets:
        for v in t:
            vert_deg[v] += 1
    # Average vertex degree per tet
    tet_avg_deg = np.array([np.mean([vert_deg[v] for v in tri.tets[i]])
                            for i in range(tri.n_tets)])

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # PR vs t
    ax = axes[0, 0]
    ax.plot(t_arr, pr, 'b-', linewidth=0.5)
    ax.axhline(nsites, color='r', linestyle='--', alpha=0.5, label=f'N={nsites}')
    ax.axhline(nsites/3, color='orange', linestyle='--', alpha=0.5, label=f'N/3={nsites//3}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('Wavefunction Spreading')
    ax.legend(fontsize=8)

    # Return probability vs t
    ax = axes[0, 1]
    ax.semilogy(t_arr, ret_prob, 'g-', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Return Probability')
    ax.set_title('Return to Origin')

    # Tet probability histogram
    ax = axes[1, 0]
    uniform = 1.0 / tri.n_tets
    ax.hist(tet_probs, bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.axvline(uniform, color='r', linestyle='--', linewidth=1.5,
               label=f'Uniform = {uniform:.4f}')
    ax.set_xlabel('Probability per tet')
    ax.set_ylabel('Count')
    ax.set_title(f'Tet Probability Distribution (t={n_steps})')
    ax.legend(fontsize=8)
    # Print stats
    print(f"\n  Tet probs: min={tet_probs.min():.6f} max={tet_probs.max():.6f} "
          f"mean={tet_probs.mean():.6f} std={tet_probs.std():.6f} "
          f"(uniform={uniform:.6f})")

    # Tet probability vs average vertex degree
    ax = axes[1, 1]
    ax.scatter(tet_avg_deg, tet_probs, s=8, alpha=0.5, c='steelblue')
    ax.axhline(uniform, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Average vertex degree of tet')
    ax.set_ylabel('Probability')
    ax.set_title('Probability vs Local Geometry')

    fig.suptitle(f'Manifold Walk: {tri.n_tets} tets, {nsites} sites, φ={mix_phi}',
                 fontsize=12)
    plt.tight_layout()
    out_path = '/tmp/manifold_walk_spreading.png'
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved to {out_path}")


if __name__ == '__main__':
    main()
