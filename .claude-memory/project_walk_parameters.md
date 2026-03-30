---
name: project_walk_parameters
description: Independent walk parameters we can vary and their possible settings, plus known results for each combination
type: project
---

## Independent Walk Parameters (1D helix chain)

### 1. Coin operator (mass term)
- (a) **β coin**: `C = cos(θ)I - i sin(θ)β` — position-independent, standard Dirac mass
- (b) **e·α coin**: `C = cos(θ)I - i sin(θ)(e·α)` — position-dependent, geometry-coupled
- (c) **No coin**: `C = I` (θ=0)

### 2. ν sign in τ = ±(√7/4)β + (3/4)(e·α)
- (a) **Constant +ν**: same sign at all sites, all steps
- (b) **Alternating per site**: +ν at even sites, -ν at odd sites
- (c) **Alternating per step**: +ν on even steps, -ν on odd steps (all sites same within a step)

### 3. Initial condition (4-component spinor × Gaussian envelope)
- (a) **(1,0,0,0)** — upper spin-up
- (b) **(1,0,1,0)/√2** — equal upper+lower
- (c) **P+/P- symmetric** — equal superposition of local tau eigenstates (site-dependent)

### 4. Frame transport
- (a) **Parallel transport**: U = (I + τ_to·τ_from)/(2cos(φ/2)) — no P+/P- mixing
- (b) **With V mixing**: V = cos(φ)I + i sin(φ)M applied after shift

## Known Results Summary

| Config | Symmetric? | Massless Dirac match? | Mass gap? |
|--------|-----------|----------------------|-----------|
| 1a + 2a + 3a (β, const+ν, (1,0,0,0)) | No (83/17) | Yes (exact) | Yes |
| 1a + 2a + 3c (β, const+ν, P+/P- sym) | Yes | Yes (exact) | Yes |
| 1b + 2a + 3a (e·α, const+ν, (1,0,0,0)) | No | Yes (exact at θ=0) | Yes |
| 1a + 2b + 3a (β, alt/site, (1,0,0,0)) | Yes (exact) | No | No (gap destroyed) |
| 1b + 2b + 3a (e·α, alt/site, (1,0,0,0)) | Yes (exact) | No | No |
| 1a + 2c + 3a (β, alt/step, (1,0,0,0)) | Yes (~0) | No | Yes (stronger) |
| 4b (V mixing at φ=π/4) | 50/50 P+/P- | — | No (alone) |

## Key Findings
- **Exact massless Dirac match**: const +ν with either (1,0,0,0) or P+/P- IC
- **(1,0,0,0)^T** projects 83% P+, 17% P- due to the β component of τ
- Alternating ν per site destroys the mass gap
- Alternating ν per step gives symmetry + mass but breaks massless Dirac match
- Best combo so far: **const +ν, P+/P- sym IC, β coin** — all three properties

## Command Line
`./walk_1d N theta sigma n_steps ic_type coin_type nu_type`
- ic_type: 0=(1,0,0,0), 1=(1,0,1,0)/√2, 2=P+/P- symmetric
- coin_type: 0=beta, 1=e·alpha
- nu_type: 0=constant +ν, 1=alternating ±ν per step

**How to apply:** Use this table to avoid re-testing known combinations. Focus on finding a configuration that achieves all three: symmetry, massless match, and mass gap.

## Dimensionless Parameter Constraints for k₀ Dispersion Measurements
1. **k₀·σ >> 1** — ensures the momentum kick winds many phase cycles across the wavepacket
2. **φ·σ = C >> 1** — ensures sufficient L/R walker mixing for Dirac-like behavior
Both must be satisfied for group velocity measurements via k₀ sweeps to be meaningful.
