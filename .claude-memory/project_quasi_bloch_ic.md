---
name: Walk-Dirac comparison and IC results
description: Best demonstration: 3.6% RMS relative error at φ_mix=0.10, σ=20, t=400; mass ratio m=0.878×φ_mix; P±-sym transported IC is best
type: project
---

## Best walk-Dirac comparison (publication figure in docs/)
Parameters: φ_mix=0.10, σ=20, m_Dirac=0.0877, t=400, N=1200, P±-sym IC
- RMS relative error: 3.6%, max relative error: 14%
- Walk tracks two-peak splitting, valley oscillations, secondary bumps
- Use relative RMS error (not L1) for fair comparison across parameters

## Mass mapping
Effective Dirac mass **m = 0.878 × φ_mix** (consistent across φ_mix=0.03–0.2)
- Predicted by quasi-Bloch W(k=0) eigenvalue
- Confirmed by density-fit mass sweep
- Speed of light c ≈ 1.0 (no correction needed)

## IC comparison
- **P±-sym transported**: best for Dirac comparison (smooth, low drift)
- β-sym transported: decent but drifts more
- Quasi-Bloch dressed: minimizes drift but high-freq oscillations ruin density match
- The P±-sym IC approximates the m=0 (smooth) component of the walk eigenstate

## Parameter sweet spots
- φ_mix = 0.05–0.12 gives best agreement (lattice artifacts small, enough mass for structure)
- Smaller σ gives more interesting Dirac structure but similar relative error (~3-4%)
- Relative error is roughly constant across σ and t; raw L1 varies with peak amplitude

**How to apply:** For walk-Dirac comparison, use P±-sym transported IC with m_Dirac = 0.878 × φ_mix. Report RMS relative error (computed where ρ_Dirac > 5% of peak). Publication figure: docs/walk_vs_dirac_publication.{png,pdf}. Script: scripts/quasi_bloch_l1.py.
