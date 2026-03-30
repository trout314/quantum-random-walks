---
name: U(2)×U(2) gauge, effective mass, and IC basis mismatch
description: Walk drift from IC basis mismatch and wrong mass; effective mass 0.0265 not 0.03; low-energy IC nearly drift-free
type: project
---

## Summary of 1D walk analysis

### Effective mass
The walk's effective mass is **m_eff = 0.0265**, not mixPhi=0.03. Using the correct mass in the Dirac comparison reduces L1 from 0.148 to **0.070**. The mass ratio m_eff/mixPhi ≈ 0.88 matches the ~0.9 factor seen in the dispersion relation.

### IC basis mismatch
The walk operates in a rotated frame (τ-basis) vs the Dirac equation (β-basis). The IC (1,0,1,0)/√2 is β-symmetric but projects asymmetrically onto the walk's eigenstates (67% pos-k, 5% neg-k), causing drift. The walk's individual eigenstates have zero velocity (10⁻¹² level) — drift is purely from projection asymmetry.

### Low-energy eigenstate IC
Constructing a localized wavepacket from near-gap (|E|<0.08) eigenstates gives:
- Smooth Gaussian envelope, σ≈49
- Walk-optimal spinor ≈ (0.535, 0.506, 0.304, 0.596) at center
- P+(τ) = 0.77 at every site (invariant)
- Drift: 3.6 sites in 80 steps (vs ~6.4 for β-symmetric)
- Spinor is 99.9% frame-transported between adjacent sites
- Shows correct massive Dirac physics (bimodal spreading)

### U(2)×U(2) gauge results
Uniform gauge phases cannot improve L1 — φ=0 is already optimal. The drift is intrinsic to the IC choice, not fixable by modifying the frame transport. Face-dependent (mod 4) gauges also don't help (local geometry is uniform: tau angle = 0.625 at every site).

### Scaling behavior
Under proper continuum limit scaling (m·σ=const, t∝σ):
- L1 is CONSTANT (0.148) — doesn't improve with resolution
- Drift = -0.136σ sites (proportional to σ)
- This is the IC basis mismatch, not a lattice artifact
- Using correct mass m=0.0265 instead of 0.03 eliminates half the error

### BC helix geometry
- Vertices at (r cos nθ, r sin nθ, nh) with r=3√3/10, θ=arccos(-2/3), h=1/√10
- Sites at barycenters of 4 consecutive vertices; scale ratio walk/BC = √(8/3)
- Chain is a quasicrystal: no translational symmetry, Bloch's theorem inapplicable
