---
name: project_1d_helix_results
description: Summary of all 1D helix walk observations — dispersion, symmetry, frame transport mixing, mass mechanism
type: project
---

## 1D Helix Walk Results

### Setup
- Single BC helix chain with R-pattern {1,3,0,2}, period-4 face sequence
- Walk operator: W = V · S · C where S = shift (P+ fwd, P- bwd with frame transport), C = coin, V = optional mixing
- τ = (√7/4)β + (3/4)(e·α) at each site, P± = (I±τ)/2
- Frame transport U = (I + τ_to·τ_from)/(2cos(φ/2)) maps P+ eigenstates perfectly (100% transmission, 0% mixing between P+ and P-)

### Key Observations

**Initial condition asymmetry:**
- Spinor (1,0,0,0)^T projects 83% into P+ and 17% into P- → asymmetric left/right peaks
- This is because τ has a large β component (√7/4 ≈ 0.661)
- Fixed by using equal P+/P- superposition as initial condition, OR by symmetric frame transport

**Speed vs θ (no mixing, original transport):**
- θ=0: v=1.0 (massless, two sharp peaks)
- θ=0.1-0.5: v≈0.78-0.80 (plateau)
- θ=0.8-1.5: v decreases to 0.54
- Roughly follows cos(θ) but with deviations
- Walk peaks are MUCH sharper (less dispersive) than Dirac prediction at all θ

**Coin mixes P+ ↔ P-:**
- θ=0: 0% mixing (identity coin)
- θ=0.5: 10% mixing per step
- θ=1.0: 31% mixing
- Mixing rate is symmetric (same P+→P- as P-→P+)

**Symmetric frame transport (V mixing):**
- V = cos(φ)I + i·sin(φ)M where M swaps P+ ↔ P- eigenspaces
- M constructed from tau eigenvectors: M = Σ_j (|pm_j⟩⟨pp_j| + |pp_j⟩⟨pm_j|)
- M anticommutes with τ: {M,τ}=0
- φ=π/4 (45°) gives exactly 50/50 splitting for ANY input spinor
- Must be applied as SEPARATE unitary step after shift (not baked into shift blocks — that breaks unitarity)
- φ=0: massless, v=1.0
- φ=0.2 (11.5°): mass gap opens, central peak + symmetric satellite peaks
- φ=π/4 (45°): large mass, most probability stays central

**Dispersion relation (V-mixing walk, verified 2026-03-25):**
- At φ=0: linear bands E=±ck (massless Dirac), c=1 site/step
- At φ>0: gap opens in ALL bands — spectrum is fully gapped, NO gapless modes
- Spectral gap = 0.855·φ (linear in φ, from direct diagonalization of walk operator on N=500 chain)
- Leading edge propagates at exactly c=1.0 for φ≤0.12 (Test B)
- Spreading speed v(C) qualitatively matches Dirac 1/√(1+C²) but fitted mass disagrees with spectral gap (IC asymmetry artifact)
- Earlier note about "four gapless bands" was from dual-parity coin experiments, NOT V-mixing

**Momentum eigenstates and gauge invariance:**
- The walk is gauge-invariant: multiplying spinors by e^{ik₀x} (with or without frame transport) has NO effect on dynamics
- The frame transport U_{i→i+1} has a constant rotation angle (25.66°) at every site, but the axis varies aperiodically
- There are NO exact momentum eigenstates because the chain has no translational symmetry (irrational screw angle)
- The walk has no separate "momentum" quantum number — propagation is characterized solely by the envelope spreading rate v(σ)
- This is a deep consequence of the local unitary structure: site-dependent phases consistent with frame transport factor out of the shift and coin operators

**Post-shift V mixing (breakthrough result):**
- V = cos(φ)I + i sin(φ)M applied AFTER the shift, where M swaps P+↔P- eigenspaces
- Without V: P+ and P- are always orthogonal at each site — no quantum interference possible, momentum kicks have no effect
- With V: enables genuine quantum interference between left-movers and right-movers
- V mixing alone (no coin, θ=0) produces mass, symmetry, AND quantum interference
- v(C) with C=φ·σ matches Dirac formula c²/√(c²+C²) with c=0.856 — much better than dual parity coin
- Scaling CV < 0.1% for C ≤ 5 across σ=100-500
- This is the most promising mass mechanism found

**Dual parity coin scaling (earlier result, superseded by V mixing):**
- v depends on the dimensionless product C = θ·σ, not on θ and σ independently
- v(C) ≈ c/√(c²+C²) with c ≈ 0.619 (qualitatively right but 30%+ error at large C)
- Scaling verified to CV < 1% for C ≤ 5 across σ = 200-2000
- Dual parity coin does NOT enable quantum interference (P+/P- always orthogonal)

**Direct dispersion measurement via momentum kicks:**
- With V mixing, k0 phase gradient produces directional propagation
- v_drift(k0) measured at small C=φσ matches Dirac formula c²k/√(c²k²+m²) excellently
- Smooth curves at small C with no lattice artifacts — true continuum behavior

**Full wavepacket shape comparison (strongest result):**
- Compared walk density profile with 1D FFT Dirac solver across C = φσ from 0.1 to 2.0
- walk_1d.c now outputs P+/P- projections (columns 4,5 in density file)
- Optimized c, m, and Dirac IC angle α simultaneously
- Results at σ=200, t=1200:
  - c = 1.000 ± 0.004 — walk speed is exactly 1 site per step
  - m = 0.9φ (linear mapping, coefficient ~0.90-0.92)
  - IC angle α = 19.5° → chi = (0.943, 0.334), 82% right-mover (matches walk's 83% P+ from (1,0,0,0))
  - RMS = 0.5% at C=0.1, < 4% for all C ≤ 2.0
- Earlier c values (0.856 from spreading speed, 0.651 from momentum kicks) were artifacts of fitting different observables without accounting for IC asymmetry; the true c=1.0
- P+ component maps to Dirac |ψ₁|², P- maps to |ψ₂|² at α=19.5°

**Open questions:**
- Extension to 3D: does V mixing work there too?
- Can we improve the IC to get closer to 50/50 P+/P- for cleaner comparison?
- Higher-order corrections: why does m/φ = 0.9 not exactly 1?

**How to apply:** The 1D walk is the cleanest test bed. Use the φ·σ = constant scaling for continuum limit comparisons. σ=200 is efficient and reliable for C ≤ 10. For Dirac comparison: c=1.0, m=0.9φ, IC angle α=19.5°.
