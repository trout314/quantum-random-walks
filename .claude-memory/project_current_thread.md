---
name: Current work thread
description: Effective mass discovery and low-energy eigenstate IC; walk matches Dirac qualitatively
type: project
---

**Where we left off:** Comparing walk evolution using low-energy eigenstate IC against Dirac solver on N=800 chain. Both show massive Dirac particle physics (bimodal spreading) but walk spreads slightly slower.

**Key discoveries this session:**

1. **Effective mass m=0.0265** (not mixPhi=0.03). Using correct mass in Dirac solver cuts L1 from 0.148 to 0.070 for β-symmetric IC. This was ~half the error.

2. **Low-energy eigenstate IC** from diagonalizing walk operator:
   - Built from |E|<0.08 eigenstates, Gaussian-overlap weighted
   - Produces smooth Gaussian wavepacket, width≈49, minimal drift (3.6 sites in 80 steps)
   - Walk-optimal spinor at center ≈ (0.535, 0.506, 0.304, 0.596)
   - P+(τ)=0.77 at every site (invariant); P+(β)≈0.55 (varies)
   - Adjacent sites' optimal spinors related by frame transport (99.9% overlap)

3. **IC basis mismatch is the primary drift source:**
   - (1,0,1,0)/√2 projects 67% pos-k, 5% neg-k onto walk eigenstates
   - Walk's individual eigenstates have zero velocity (10⁻¹² level)
   - L1=0.148 constant under scaling (m·σ=const) → not a lattice artifact
   - Frame transport on IC helps: transported ICs drift ±6 vs ±40 untransported

4. **Walk and Dirac show same qualitative physics** (massive particle bimodal spreading) but different spreading rates due to basis/mass differences.

**Open questions for next session:**
1. Can we fit the Dirac mass to match the low-energy IC's spreading rate? (Need to try larger masses)
2. What determines the walk-optimal spinor from local geometry? (P+(τ)=0.77 is invariant — why?)
3. Can we derive m_eff = 0.0265 from mixPhi = 0.03 analytically?
4. How does this all connect to the 3D walk drift?

**Tools:**
- `scripts/fourier_dispersion.py`: build_chain, build_walk_operator, diagonalize
- `scripts/build_optimal_ic.py`: construct and save low-energy eigenstate IC
- `scripts/test_1d_drift.py`: L1 measurement
- D walk: icType=3 (file IC), icType=4 (walk-optimal transported)
- Diagonalization: N=800 takes ~50s, N=400 takes ~4s
