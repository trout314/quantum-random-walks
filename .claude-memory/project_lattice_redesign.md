---
name: 3D lattice redesign — R/L chains cross, perpendicular axes (FIXED but tree problem)
description: R/L chains cross at single sites with perpendicular axes. Implemented correctly. But flat-space lattice is a tree → no interference → norm leaks. Need closed manifold.
type: project
---

## Lattice structure (implemented correctly)
- Each site on exactly one R-chain and one L-chain (crossing point)
- R-chain uses edge {v_n, v_{n+3}}, L-chain uses perpendicular edge {v_{n+1}, v_{n+2}}
- R_edge · L_edge = 0 exactly at every site (verified)
- L-chain vertex sequence = permutation [1,3,0,2] of R-chain vertices
- Chains share at most 1 site, then diverge completely

## Seed generation (working)
- Site-orbit frontier with priority queue (distance from origin)
- A4 symmetric: extend canonical site, find matching extensions at 11 partners
- No chain orbits — each site's chains are independent
- Verified: correct counts, zero orphans, isotropy passes

## The tree problem (fundamental)
In flat R³, chains diverge exponentially and never reconverge:
- Lattice is a tree graph — no loops
- Short chain segments everywhere → internal absorbing boundaries
- Runtime extension creates exponential site growth (~2× per step)
- Norm drops to 0 in ~10 steps regardless of thresholds
- The only interference is forward/backward along a single chain

## Resolution
Move to triangulated closed 3-manifold where chains loop back.
See project_manifold_goal.md for details.
