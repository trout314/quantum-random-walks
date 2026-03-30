---
name: project_bc_helix_walk
description: Walk scheme based on Boerdijk-Coxeter helix decomposition into disjoint L-spirals and R-spirals
type: project
---

The walk uses Boerdijk-Coxeter (BC) helix structure:
- Walker steps go from one tet center through a face to the next tet center
- A BC helix path cycles through 3 of 4 step directions; the two cyclic orderings give L and R chirality
- All L-spirals are **disjoint** (no shared sites), and all R-spirals are disjoint
- But L-spirals and R-spirals DO intersect (share sites)
- Walk operator: W = S_R · S_L, where S_L shifts along all L-spirals at once and S_R shifts along all R-spirals at once
- Each shift is unitary (independent 1D shifts along non-overlapping chains)
- Analogous to cubic lattice W = S_z · S_y · S_x

**Why:** This reduces the 3D tetrahedral walk to a pair of 1D walk shifts, each manifestly unitary.

**How to apply:** This supersedes the generic approaches in project_shift_operator_approaches.md (though those are still worth noting). The BCH analysis for the continuum limit of S_R · S_L is the next step.
