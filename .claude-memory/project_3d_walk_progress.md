---
name: 3D walk progress — lattice correct but flat-space tree problem blocks dynamics
description: Lattice structure correct (perpendicular R/L crossings). Balanced IC (1,0,i,0)/√2 isotropic. But flat-space tree topology causes exponential growth and norm leak. Need closed manifold.
type: project
---

## What works
- Lattice geometry: perpendicular R/L chains, A4 symmetric seed, correct crossings
- Balanced IC: ψ=(1,0,i,0)/√2, <β>=<α_i>=<τ_a>=0, isotropic spreading
- 1D walk: proven unitary, 3.6% RMS match to Dirac equation
- Walk operators (shift, Vmix, frame transport): all individually correct
- Analytic BC helix geometry: no reflections needed, everything from vertex formula

## What doesn't work
- Flat-space 3D walk: norm drops to 0 in ~10 steps
- Exponential site growth (~2× per step) from chain branching
- Internal absorbing boundaries (short chains throughout interior)
- Root cause: flat-space lattice is a tree, no path reconvergence, no interference

## Key findings
- The shift operator IS unitary on individual chains (verified)
- Open-boundary cross-terms cause norm² + absorbed > 1.0 (not a code bug)
- With runtime extension + cross-chain creation, performance is workable
- But the tree topology makes the walk physically wrong, not just numerically

## Next: closed 3-manifold
Move to triangulated closed 3-manifold where chains loop back.
The walk construction (τ, frame transport, shift, Vmix) is local and generalizes.
See project_manifold_goal.md.
