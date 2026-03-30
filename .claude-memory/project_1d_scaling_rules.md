---
name: 1D walk scaling rules for Dirac correspondence
description: Parameter hierarchy 1/N << φ << 1 << σ << N; mass m=0.878φ; seven conditions for walk to match Dirac equation
type: project
---

The walk reproduces the Dirac equation when: **1/N << φ_mix << 1 << σ << N**

Seven conditions (all must hold):
1. **σ >> 1** — continuum spatial resolution (suppresses lattice artifacts)
2. **φ·σ >> 1** — mass gap resolved (wavepacket spans many Compton wavelengths)
3. **k₀·σ >> 1** — momentum kick well-defined (for kicked wavepackets)
4. **φ << 1** — Dirac regime (lattice dispersion corrections small; degrades above ~0.2)
5. **N >> σ + t** — boundary avoidance
6. **t >> mσ²** — splitting time (for Zitterbewegung structure to develop)
7. **φN >> 1** — spectral resolution (mass gap visible in eigenvalue spectrum)

**Mass mapping:** m = 0.878 × φ_mix (not φ_mix itself)

**Best parameters:** φ=0.10, σ=20, t=400, N=1200 → 3.6% RMS relative error
**IC:** P±-symmetric frame-transported

**Why:** These rules define the parameter regime where the walk is a faithful Dirac simulator. Violating any condition degrades agreement in a specific, predictable way.

**How to apply:** When choosing parameters for any 1D walk experiment or comparison, verify all seven conditions are satisfied. The splitting time condition t >> mσ² is often the binding constraint — it requires t ∝ σ² for fixed mass, making large-σ runs expensive.

Full writeup: docs/1d_walk_scaling_rules.md
