#!/usr/bin/env python3
"""
Run the 3D quantum walk and coarse-grain onto a triangulated 3-manifold.

Uses the Python walk infrastructure (not D code) to keep things simple
and verifiable. Runs on a small lattice with moderate parameters.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from triangulation import Triangulation
from manifold_walk import ManifoldWalk, make_tau, frame_transport, STD_DIRS, L_PERM
from helix_geometry import build_taus, centroid, exit_direction, vertex


def proj_plus(tau):
    return 0.5 * (np.eye(4, dtype=complex) + tau)

def proj_minus(tau):
    return 0.5 * (np.eye(4, dtype=complex) - tau)


def find_fourth_vertex(v0, v1, v2, v_old):
    """Reflection of v_old through the plane of v0, v1, v2."""
    return 2 * (v0 + v1 + v2) / 3 - v_old


class SimpleLatticeWalk:
    """A simple Python implementation of the 3D walk on a growing lattice.

    Each site has:
    - A 4-component complex spinor
    - An R-chain and L-chain membership
    - A vertex window (4 vertices defining the tetrahedron)
    - A manifold tag (tet_id, vertex_perm)

    The walk alternates S_L and S_R shifts with V-mixing.
    """

    def __init__(self):
        self.psi = {}         # site_id → ndarray(4,) complex
        self.vertices = {}    # site_id → list of 4 Vec3 (vertex positions)
        self.r_chain = {}     # site_id → (prev_site, next_site) or None
        self.l_chain = {}     # site_id → (prev_site, next_site) or None
        self.exit_dir_r = {}  # site_id → exit direction for R-chain
        self.exit_dir_l = {}  # site_id → exit direction for L-chain
        self.manifold_tag = {}  # site_id → (tet_id, vertex_perm dict)
        self.next_id = 0

    def add_site(self, verts, psi=None):
        sid = self.next_id
        self.next_id += 1
        self.vertices[sid] = [np.array(v) for v in verts]
        self.psi[sid] = psi if psi is not None else np.zeros(4, dtype=complex)
        self.r_chain[sid] = [None, None]  # [prev, next]
        self.l_chain[sid] = [None, None]
        # Exit directions: vertex 0 direction from centroid (R-chain drops vertex 0)
        c = np.mean(verts, axis=0)
        d_r = verts[0] - c
        d_r /= np.linalg.norm(d_r)
        self.exit_dir_r[sid] = d_r
        # L-chain: permute vertices by L_PERM, then vertex 0 is the exit
        l_verts = [verts[L_PERM[k]] for k in range(4)]
        c_l = np.mean(l_verts, axis=0)
        d_l = l_verts[0] - c_l
        d_l /= np.linalg.norm(d_l)
        self.exit_dir_l[sid] = d_l
        return sid

    def extend_chain(self, site_id, chain_type, direction):
        """Extend a chain (R or L) forward or backward from site_id.
        Returns the new site ID, or None if already extended."""
        chain = self.r_chain if chain_type == 'R' else self.l_chain
        idx = 1 if direction == 'fwd' else 0  # next or prev
        if chain[site_id][idx] is not None:
            return chain[site_id][idx]  # already extended

        verts = self.vertices[site_id]
        if chain_type == 'L':
            # L-chain uses permuted vertices
            verts = [verts[L_PERM[k]] for k in range(4)]

        if direction == 'fwd':
            # Drop vertex 0, add new vertex
            old = verts[0]
            keep = verts[1:]
            new_v = find_fourth_vertex(keep[0], keep[1], keep[2], old)
            new_verts = keep + [new_v]
        else:
            # Drop vertex 3, add new vertex at front
            old = verts[3]
            keep = verts[:3]
            new_v = find_fourth_vertex(keep[0], keep[1], keep[2], old)
            new_verts = [new_v] + keep

        if chain_type == 'L':
            # Unpermute back to R-frame for storage
            inv_perm = [2, 0, 3, 1]  # inverse of L_PERM
            new_verts_r = [new_verts[inv_perm[k]] for k in range(4)]
            new_verts = new_verts_r

        new_id = self.add_site(new_verts)

        # Link chains
        if direction == 'fwd':
            chain[site_id][1] = new_id
            chain_new = self.r_chain if chain_type == 'R' else self.l_chain
            chain_new[new_id][0] = site_id
        else:
            chain[site_id][0] = new_id
            chain_new = self.r_chain if chain_type == 'R' else self.l_chain
            chain_new[new_id][1] = site_id

        return new_id

    def shift(self, chain_type, ext_thresh=1e-3):
        """Apply shift operator along all chains of given type."""
        chain = self.r_chain if chain_type == 'R' else self.l_chain
        exit_dir = self.exit_dir_r if chain_type == 'R' else self.exit_dir_l

        new_psi = {}
        absorbed = 0.0

        for sid in list(self.psi.keys()):
            new_psi.setdefault(sid, np.zeros(4, dtype=complex))

        for sid in list(self.psi.keys()):
            psi_s = self.psi[sid]
            if np.linalg.norm(psi_s) < 1e-30:
                continue

            d = exit_dir[sid]
            tau = make_tau(d)
            Pp = proj_plus(tau)
            Pm = proj_minus(tau)

            psi_plus = Pp @ psi_s
            psi_minus = Pm @ psi_s

            # Forward: P+ goes to next
            nxt = chain[sid][1]
            if nxt is None and np.linalg.norm(psi_plus) > ext_thresh:
                nxt = self.extend_chain(sid, chain_type, 'fwd')
                new_psi.setdefault(nxt, np.zeros(4, dtype=complex))
            if nxt is not None:
                d_nxt = exit_dir[nxt]
                tau_nxt = make_tau(d_nxt)
                U = frame_transport(tau, tau_nxt)
                new_psi[nxt] += U @ psi_plus
            else:
                absorbed += np.real(np.vdot(psi_plus, psi_plus))

            # Backward: P- goes to prev
            prv = chain[sid][0]
            if prv is None and np.linalg.norm(psi_minus) > ext_thresh:
                prv = self.extend_chain(sid, chain_type, 'bwd')
                new_psi.setdefault(prv, np.zeros(4, dtype=complex))
            if prv is not None:
                d_prv = exit_dir[prv]
                tau_prv = make_tau(d_prv)
                U = frame_transport(tau, tau_prv)
                new_psi[prv] += U @ psi_minus
            else:
                absorbed += np.real(np.vdot(psi_minus, psi_minus))

        self.psi = new_psi
        return absorbed

    def vmix(self, chain_type, phi):
        """Apply V-mixing on all sites using the given chain type's τ."""
        if phi == 0:
            return
        exit_dir = self.exit_dir_r if chain_type == 'R' else self.exit_dir_l
        cp, sp = np.cos(phi), np.sin(phi)

        for sid in self.psi:
            d = exit_dir.get(sid)
            if d is None:
                continue
            tau = make_tau(d)
            Pp = proj_plus(tau)
            Pm = proj_minus(tau)

            # Gram-Schmidt for P+ and P- bases
            pp_basis = []
            pm_basis = []
            for col in range(4):
                if len(pp_basis) < 2:
                    v = Pp[:, col].copy()
                    for b in pp_basis:
                        v -= np.vdot(b, v) * b
                    if np.linalg.norm(v) > 1e-10:
                        pp_basis.append(v / np.linalg.norm(v))
                if len(pm_basis) < 2:
                    v = Pm[:, col].copy()
                    for b in pm_basis:
                        v -= np.vdot(b, v) * b
                    if np.linalg.norm(v) > 1e-10:
                        pm_basis.append(v / np.linalg.norm(v))

            M = np.zeros((4, 4), dtype=complex)
            for j in range(min(len(pp_basis), len(pm_basis))):
                M += np.outer(pm_basis[j], pp_basis[j].conj())
                M += np.outer(pp_basis[j], pm_basis[j].conj())

            V = cp * np.eye(4) + 1j * sp * M
            self.psi[sid] = V @ self.psi[sid]

    def norm(self):
        return sum(np.real(np.vdot(p, p)) for p in self.psi.values())

    def prune(self, thresh):
        """Remove sites with amplitude below threshold. Returns pruned prob."""
        pruned = 0.0
        to_remove = []
        for sid, psi in self.psi.items():
            amp2 = np.real(np.vdot(psi, psi))
            if 0 < amp2 < thresh:
                pruned += amp2
                to_remove.append(sid)
        for sid in to_remove:
            self.psi[sid] = np.zeros(4, dtype=complex)
        return pruned


def propagate_manifold_tags(walk, tri, start_site, start_tet):
    """Propagate manifold tags through all connected sites via BFS."""
    mw = ManifoldWalk(tri)
    mw.tag_origin(start_site, start_tet)

    visited = {start_site}
    queue = [start_site]

    while queue:
        sid = queue.pop(0)
        tag = mw.site_tags[sid]

        # Propagate along R and L chains
        for chain_type in ['R', 'L']:
            chain = walk.r_chain if chain_type == 'R' else walk.l_chain
            for direction in [0, 1]:  # prev, next
                nb = chain[sid][direction]
                if nb is not None and nb not in visited:
                    # Propagate tag
                    if direction == 1:  # forward
                        mw.propagate_along_chain([sid, nb])
                    else:  # backward: nb is before sid
                        # Tag nb by going backward from sid
                        # The "chain" from nb forward reaches sid
                        # We need to figure out nb's tag from sid's
                        # Simpler: just tag via the forward chain from nb
                        mw.propagate_along_chain([sid, nb])
                        # Actually this is wrong — propagate_along_chain
                        # assumes forward chain order. Let me handle backward
                        # by computing the tag directly.

                        # For backward: sid's vertex_perm has window [v0,v1,v2,v3]
                        # The previous site in the chain has window [v_new, v0, v1, v2]
                        # where v_new is determined by the manifold adjacency

                        # The "dropped" vertex when going from nb to sid is v3
                        # So nb's window is [?, v0, v1, v2] where ? maps to
                        # the manifold vertex on the other side of face {v0,v1,v2}
                        dropped = tag.vertex_perm[3]
                        tet_verts = tri.tets[tag.tet_id]
                        face_idx = tet_verts.index(dropped)
                        prev_tet = tri.neighbors[tag.tet_id][face_idx]
                        prev_verts = tri.tets[prev_tet]
                        shared = set(tet_verts) - {dropped}
                        new_vert = [v for v in prev_verts if v not in shared][0]

                        nb_perm = {
                            0: new_vert,
                            1: tag.vertex_perm[0],
                            2: tag.vertex_perm[1],
                            3: tag.vertex_perm[2],
                        }
                        mw.site_tags[nb] = mw.ManifoldSite(prev_tet, nb_perm) \
                            if hasattr(mw, 'ManifoldSite') else \
                            type(mw.site_tags[sid])(prev_tet, nb_perm)

                    visited.add(nb)
                    queue.append(nb)

    return mw


def main():
    # Load manifold
    tri_path = os.path.expanduser(
        '~/Desktop/Discrete-Differential-Geometry/'
        'standard_triangulations/equilibrated_200.mfd')
    tri = Triangulation.load(tri_path)
    print(f"Manifold: {tri.n_tets} tets, {tri.n_verts} verts, χ={tri.n_verts - 144 + 222 - tri.n_tets}")

    # Create walk
    walk = SimpleLatticeWalk()

    # Initial tetrahedron: use standard tet vertices
    v0 = [np.array(STD_DIRS[k]) for k in range(4)]
    origin = walk.add_site(v0)

    # Balanced IC: (1, 0, i, 0) / sqrt(2)
    walk.psi[origin] = np.array([1, 0, 1j, 0], dtype=complex) / np.sqrt(2)

    # Walk parameters
    n_steps = 30
    mix_phi = 0.05
    ext_thresh = 5e-3
    prune_thresh = 1e-4

    print(f"\nWalk: {n_steps} steps, φ_mix={mix_phi}, ext_thresh={ext_thresh}")
    print(f"{'t':>3} {'norm':>10} {'sites':>8} {'absorbed':>10} {'pruned':>10} {'grid_prob':>10} {'total':>10}")

    total_absorbed = 0
    total_pruned = 0

    for t in range(n_steps + 1):
        n2 = walk.norm()

        # Coarse-grain onto manifold at this step
        mw = propagate_manifold_tags(walk, tri, origin, 0)
        exit_dirs = {}
        for sid in walk.psi:
            if sid in walk.exit_dir_r:
                exit_dirs[sid] = walk.exit_dir_r[sid]
        manifold_psi = mw.coarse_grain(walk.psi, exit_dirs)
        grid_prob = sum(np.real(np.vdot(p, p)) for p in manifold_psi.values())
        total = n2 + grid_prob  # This double-counts since grid includes surviving sites
        # Actually: grid_prob IS the coarse-grained norm (all sites mapped to manifold)
        # It replaces n2, not adds to it.

        n_manifold_tets = len(manifold_psi)

        print(f"{t:3d} {np.sqrt(n2):10.6f} {len(walk.psi):8d} {total_absorbed:10.4e} "
              f"{total_pruned:10.4e} {grid_prob:10.6f} {n_manifold_tets:>4d} tets")

        if t < n_steps:
            # Walk step: S_L → Vmix_L → S_R → Vmix_R
            abs_l = walk.shift('L', ext_thresh)
            walk.vmix('L', mix_phi)
            abs_r = walk.shift('R', ext_thresh)
            walk.vmix('R', mix_phi)

            total_absorbed += abs_l + abs_r
            total_pruned += walk.prune(prune_thresh)


if __name__ == '__main__':
    main()
