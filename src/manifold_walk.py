"""
Quantum walk on a triangulated 3-manifold via coarse-graining.

Runs the standard 3D BC helix walk, but each lattice site is tagged with
a manifold tetrahedron ID and vertex permutation. The coarse-grained
wavefunction is obtained by summing (with frame transport) all site
amplitudes that map to the same manifold tetrahedron.

The walk itself is unchanged — only the bookkeeping and final coarse-graining
are new.
"""

import numpy as np
from itertools import combinations
from src.triangulation import Triangulation
from src.dirac import alpha as ALPHA, beta as BETA
from src.tau_operators import NU


def make_tau(d):
    """Construct τ from a unit direction vector."""
    return NU * BETA + 0.75 * sum(d[a] * ALPHA[a] for a in range(3))


def frame_transport(tau_from, tau_to):
    """Compute frame transport unitary between two τ operators."""
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    cos_half = np.sqrt(max((1 + cos_theta) / 2, 1e-15))
    scale = 1 / (2 * cos_half)
    return scale * (np.eye(4, dtype=complex) + prod)


# Standard tetrahedral directions (unit sphere)
STD_DIRS = np.array([
    [0, 0, 1],
    [2*np.sqrt(2)/3, 0, -1/3],
    [-np.sqrt(2)/3, np.sqrt(6)/3, -1/3],
    [-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3],
])

# The vertex permutation for the cross-chain (L from R)
L_PERM = [1, 3, 0, 2]


class ManifoldSite:
    """A site in the 3D walk tagged with manifold data."""
    def __init__(self, tet_id, vertex_perm):
        """
        tet_id: which manifold tetrahedron this site corresponds to
        vertex_perm: dict mapping walk vertex label → manifold vertex label
            The walk uses a sliding window of 4 vertex labels.
            vertex_perm[k] = manifold vertex at position k in the window.
        """
        self.tet_id = tet_id
        self.vertex_perm = vertex_perm


class ManifoldWalk:
    """Manages the mapping between 3D walk sites and manifold tetrahedra."""

    def __init__(self, tri):
        """
        Parameters
        ----------
        tri : Triangulation
            The target 3-manifold triangulation.
        """
        self.tri = tri
        self.site_tags = {}  # site_id → ManifoldSite

    def tag_origin(self, site_id, manifold_tet_id):
        """Tag the origin site with a manifold tetrahedron.

        The origin tet's vertices are identified with the standard tet:
        manifold_vert[i] ↔ walk_vert[i] for i = 0,1,2,3.
        """
        tet_verts = self.tri.tets[manifold_tet_id]
        # vertex_perm[window_pos] = manifold_vertex
        perm = {k: tet_verts[k] for k in range(4)}
        self.site_tags[site_id] = ManifoldSite(manifold_tet_id, perm)

    def propagate_along_chain(self, chain_sites, chain_type='R'):
        """Propagate manifold tags along a chain of site IDs.

        At each step, the chain drops one vertex and adds a new one.
        For an R-chain, vertex 0 (the oldest) is dropped.
        For an L-chain with permutation [1,3,0,2], vertex 0 of the
        L-window is vertex 1 of the R-window, etc.

        Parameters
        ----------
        chain_sites : list of int
            Site IDs along the chain, starting from a tagged site.
        chain_type : 'R' or 'L'
        """
        if len(chain_sites) < 2:
            return

        # The first site must already be tagged
        first = chain_sites[0]
        assert first in self.site_tags, \
            f"First site {first} in chain not yet tagged"

        tag = self.site_tags[first]
        tet_id = tag.tet_id
        perm = dict(tag.vertex_perm)  # copy

        for i in range(1, len(chain_sites)):
            site_id = chain_sites[i]

            # The chain drops vertex at window position 0 (the oldest).
            # The dropped vertex is opposite the exit face.
            dropped_vert = perm[0]

            # Find which face of the manifold tet is opposite the dropped vertex
            tet_verts = self.tri.tets[tet_id]
            face_idx = tet_verts.index(dropped_vert)

            # The neighbor tet across this face
            neighbor_id = self.tri.neighbors[tet_id][face_idx]
            neighbor_verts = self.tri.tets[neighbor_id]

            # The shared face vertices (3 vertices kept)
            shared = set(tet_verts) - {dropped_vert}

            # The new vertex in the neighbor (not in the shared face)
            new_vert = [v for v in neighbor_verts if v not in shared]
            assert len(new_vert) == 1, \
                f"Expected 1 new vertex, got {len(new_vert)}"
            new_vert = new_vert[0]

            # Update the vertex permutation: shift window by 1
            # Old: perm[0]=dropped, perm[1], perm[2], perm[3]
            # New: perm[0]=old perm[1], perm[1]=old perm[2],
            #      perm[2]=old perm[3], perm[3]=new_vert
            new_perm = {
                0: perm[1],
                1: perm[2],
                2: perm[3],
                3: new_vert,
            }

            tet_id = neighbor_id
            perm = new_perm

            self.site_tags[site_id] = ManifoldSite(tet_id, perm)

    def coarse_grain(self, psi, exit_dirs):
        """Coarse-grain the wavefunction onto the manifold tetrahedra.

        For each manifold tet, collect all lattice sites mapping to it,
        frame-transport their spinors to a common reference frame,
        and sum.

        Parameters
        ----------
        psi : dict of int → ndarray(4,) complex
            Spinor amplitude at each site.
        exit_dirs : dict of int → ndarray(3,)
            Exit direction at each site (for frame transport).

        Returns
        -------
        manifold_psi : dict of int → ndarray(4,) complex
            Coarse-grained spinor at each manifold tetrahedron.
        """
        # Group sites by manifold tet
        tet_sites = {}  # manifold_tet_id → list of (site_id, exit_dir, spinor)
        for site_id, tag in self.site_tags.items():
            if site_id not in psi:
                continue
            spinor = psi[site_id]
            if np.linalg.norm(spinor) < 1e-30:
                continue
            tet_id = tag.tet_id
            if tet_id not in tet_sites:
                tet_sites[tet_id] = []
            tet_sites[tet_id].append((site_id, exit_dirs.get(site_id), spinor))

        # For each manifold tet, pick the first site's frame as reference
        # and frame-transport all others to it before summing.
        manifold_psi = {}
        for tet_id, sites in tet_sites.items():
            ref_dir = sites[0][1]
            if ref_dir is None:
                # No exit direction available, just sum raw
                total = sum(s[2] for s in sites)
                manifold_psi[tet_id] = total
                continue

            tau_ref = make_tau(ref_dir)
            total = sites[0][2].copy()  # first site: no transport needed

            for _, edir, spinor in sites[1:]:
                if edir is None:
                    total += spinor
                    continue
                tau_site = make_tau(edir)
                U = frame_transport(tau_site, tau_ref)
                total += U @ spinor

            manifold_psi[tet_id] = total

        return manifold_psi


if __name__ == '__main__':
    # Quick test: load triangulation, tag a few sites manually
    tri = Triangulation.load(
        '/home/aaron-trout/Desktop/Discrete-Differential-Geometry/'
        'standard_triangulations/equilibrated_200.mfd')
    tri.summary()

    mw = ManifoldWalk(tri)
    mw.tag_origin(0, 0)  # site 0 → manifold tet 0

    # Simulate a short R-chain: walk through face adjacency
    print("\nTracing R-chain from tet 0:")
    tet = 0
    visited = [tet]
    for step in range(20):
        # R-chain drops vertex 0 (oldest in window)
        tag = mw.site_tags[step]
        dropped = tag.vertex_perm[0]
        tet_verts = tri.tets[tag.tet_id]
        face_idx = tet_verts.index(dropped)
        next_tet = tri.neighbors[tag.tet_id][face_idx]
        visited.append(next_tet)
        # Create a fake site for the next step
        mw.propagate_along_chain([step, step + 1])

    print(f"  Tets visited: {visited}")
    print(f"  Unique tets: {len(set(visited))}")
    print(f"  Looped back: {len(visited) != len(set(visited))}")
