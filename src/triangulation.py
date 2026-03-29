"""
Triangulation loader and face-adjacency graph for closed 3-manifolds.

File format (.mfd): a comment line starting with #, followed by a single
line containing a Python-parseable list of 4-tuples [v1, v2, v3, v4],
each representing a tetrahedron by its vertex labels.
"""

import ast
import numpy as np
from collections import Counter
from itertools import combinations


class Triangulation:
    """A triangulated closed 3-manifold."""

    def __init__(self, tetrahedra):
        """
        Parameters
        ----------
        tetrahedra : list of list/tuple of 4 ints
            Each entry [v1, v2, v3, v4] is a tetrahedron.
        """
        self.tets = [tuple(sorted(t)) for t in tetrahedra]
        self.n_tets = len(self.tets)

        # Collect vertices
        verts = set()
        for t in self.tets:
            verts.update(t)
        self.vertices = sorted(verts)
        self.n_verts = len(self.vertices)

        # Build tet index: tet_id → (v0, v1, v2, v3) with v0 < v1 < v2 < v3
        # and reverse: frozenset of 4 vertices → tet_id
        self.tet_to_id = {}
        for i, t in enumerate(self.tets):
            self.tet_to_id[frozenset(t)] = i

        # Build face adjacency.
        # A face is a frozenset of 3 vertices. Each face is shared by exactly
        # 2 tetrahedra in a closed manifold.
        self._build_adjacency()

    def _build_adjacency(self):
        """Build the face-adjacency graph.

        For each tetrahedron, compute its 4 faces and find the neighboring
        tetrahedron across each face.

        Stores:
            neighbors[tet_id][face_idx] = neighbor_tet_id
            face_vertices[tet_id][face_idx] = (v1, v2, v3) — the 3 shared vertices
            opposite_vertex[tet_id][face_idx] = v — the vertex NOT on this face
        """
        # Map: face (frozenset of 3 verts) → list of (tet_id, face_idx, opposite_vertex)
        face_map = {}

        for tet_id, t in enumerate(self.tets):
            for face_idx in range(4):
                # Face face_idx is opposite to vertex t[face_idx]
                opp = t[face_idx]
                face_verts = frozenset(v for v in t if v != opp)
                if face_verts not in face_map:
                    face_map[face_verts] = []
                face_map[face_verts].append((tet_id, face_idx, opp))

        # Build neighbor arrays
        self.neighbors = [[-1] * 4 for _ in range(self.n_tets)]
        self.face_vertices = [[None] * 4 for _ in range(self.n_tets)]
        self.opposite_vertex = [[None] * 4 for _ in range(self.n_tets)]

        for face_verts, entries in face_map.items():
            if len(entries) != 2:
                raise ValueError(
                    f"Face {face_verts} has {len(entries)} tetrahedra "
                    f"(expected 2 for a closed manifold)"
                )
            (id_a, fi_a, opp_a), (id_b, fi_b, opp_b) = entries
            self.neighbors[id_a][fi_a] = id_b
            self.neighbors[id_b][fi_b] = id_a
            self.face_vertices[id_a][fi_a] = tuple(sorted(face_verts))
            self.face_vertices[id_b][fi_b] = tuple(sorted(face_verts))
            self.opposite_vertex[id_a][fi_a] = opp_a
            self.opposite_vertex[id_b][fi_b] = opp_b

        # Verify: every face has a neighbor
        for tet_id in range(self.n_tets):
            for fi in range(4):
                assert self.neighbors[tet_id][fi] >= 0, \
                    f"Tet {tet_id} face {fi} has no neighbor"

    def face_idx_to_neighbor(self, tet_id, face_idx):
        """Return the neighbor tet across the given face."""
        return self.neighbors[tet_id][face_idx]

    def shared_face(self, tet_a, tet_b):
        """Find which face index of tet_a is shared with tet_b.
        Returns (face_idx_a, face_idx_b) or None if not neighbors."""
        for fi_a in range(4):
            if self.neighbors[tet_a][fi_a] == tet_b:
                for fi_b in range(4):
                    if self.neighbors[tet_b][fi_b] == tet_a:
                        return fi_a, fi_b
        return None

    @staticmethod
    def load(filename):
        """Load a triangulation from a .mfd file."""
        with open(filename) as f:
            lines = f.readlines()
        # Skip comment lines
        data_line = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                data_line = line
                break
        if data_line is None:
            raise ValueError("No data found in file")
        tets = ast.literal_eval(data_line)
        return Triangulation(tets)

    def summary(self):
        """Print a summary of the triangulation."""
        print(f"Tetrahedra: {self.n_tets}")
        print(f"Vertices: {self.n_verts}")

        # Euler characteristic: V - E + F - T
        # For a closed 3-manifold triangulation:
        # Each tet has 6 edges, each edge shared by multiple tets
        edges = set()
        for t in self.tets:
            for e in combinations(t, 2):
                edges.add(frozenset(e))
        n_edges = len(edges)

        # Each tet has 4 faces, each face shared by 2 tets
        n_faces = 2 * self.n_tets  # each tet has 4 faces, shared pairwise

        euler = self.n_verts - n_edges + n_faces - self.n_tets
        print(f"Edges: {n_edges}")
        print(f"Faces: {n_faces}")
        print(f"Euler characteristic: χ = {euler}")
        if euler == 0:
            print("  (consistent with S³, T³, or other χ=0 manifold)")

        # Vertex degree distribution
        deg = Counter()
        for t in self.tets:
            for v in t:
                deg[v] += 1
        degs = sorted(deg.values())
        print(f"Tets per vertex: min={degs[0]}, max={degs[-1]}, "
              f"median={degs[len(degs)//2]}")

        # Check all faces have exactly 2 tets
        face_count = Counter()
        for t in self.tets:
            for face in combinations(t, 3):
                face_count[frozenset(face)] += 1
        boundary = sum(1 for c in face_count.values() if c == 1)
        print(f"Boundary faces: {boundary} (should be 0 for closed manifold)")


if __name__ == '__main__':
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/aaron-trout/Desktop/Discrete-Differential-Geometry/standard_triangulations/equilibrated_200.mfd'
    tri = Triangulation.load(fname)
    tri.summary()

    print(f"\nFirst 5 tetrahedra:")
    for i in range(min(5, tri.n_tets)):
        t = tri.tets[i]
        nbrs = [tri.neighbors[i][fi] for fi in range(4)]
        print(f"  tet {i}: verts={t}, neighbors={nbrs}")

    # Verify adjacency symmetry
    for i in range(tri.n_tets):
        for fi in range(4):
            j = tri.neighbors[i][fi]
            result = tri.shared_face(j, i)
            assert result is not None, f"Adjacency not symmetric: tet {i} → {j}"
    print(f"\nAdjacency symmetry: verified ✓")
