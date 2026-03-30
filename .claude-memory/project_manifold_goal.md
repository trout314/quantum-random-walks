---
name: Flat-space fundamental issue; 3-manifold is the path forward
description: Flat-space BC helix chains form a tree (no loops, no interference, exponential growth). Closed 3-manifold gives finite sites, reconvergence, interference, unitarity.
type: project
---

## Fundamental issue with flat-space 3D walk (discovered 2026-03-29)

BC helix chains in flat R³ diverge exponentially and NEVER reconverge. The lattice is a tree:
- No loops → no interference between different paths
- Only interference is forward/backward along a single chain
- Each chain extension creates new sites that branch in new directions
- Exponential growth: ~2× sites per step, even with aggressive pruning
- Short chain segments throughout the interior create absorbing boundaries everywhere
- Norm drops catastrophically (1.0 → 0 in ~10 steps with extension threshold 1e-2)

The lack of interference is a property of the flat-space lattice, not the walk construction. The walk operators (τ, frame transport, shift, V-mixing) are all local and work on ANY face-stacking chain of tetrahedra.

## Why the old code seemed to work

The old code had R and L chains OVERLAPPING (sharing all sites). This was geometrically wrong but meant no cross-chain explosion and no internal absorbing boundaries.

## Why a closed 3-manifold solves everything

On a triangulated 3-manifold (e.g., S³, T³):
- **Finite sites**: the triangulation has a fixed number of tetrahedra
- **Chains reconverge**: extending far enough returns to visited tetrahedra → loops
- **Loops → interference**: different paths interfere at shared sites
- **Exact unitarity**: no absorbing boundaries (every chain loops)
- **Physical geometry**: curvature encoded in the triangulation

The 1D walk works because periodic BCs make chains loop — same principle.

## Connection to earlier findings

The BC helix is periodic in the 600-cell on S³ (period 30 tetrahedra). The 600-cell triangulation of S³ would give a natural closed manifold where BC helix chains loop. This connects to the cut-and-project analysis from the quasi-Bloch work.

## Next step

Implement the walk on a triangulated closed 3-manifold. Candidates:
1. **600-cell on S³**: natural home of the BC helix, 600 tetrahedra, chains loop with period 30
2. **Flat torus T³**: periodic cubic lattice subdivided into tetrahedra
3. **Any closed triangulation**: the walk construction generalizes to arbitrary face-stacking chains

**How to apply:** The 1D walk code and operator infrastructure are correct and proven unitary. The 3D lattice structure (perpendicular R/L crossing chains, A4 symmetric seed generation) is correct. Only the lattice TOPOLOGY needs to change from tree (flat space) to graph with loops (closed manifold).
