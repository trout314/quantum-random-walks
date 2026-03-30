---
name: project_validation_tests
description: List of tests to demonstrate the walk reproduces Dirac equation behavior, for convincing a skeptical reviewer
type: project
---

## Tests for Dirac Equation Correspondence

### 1D Helix (established results)
These are done and show correct behavior:
- [x] Exact massive Dirac dispersion E² = m² + k² (R² > 0.99)
- [x] Mass proportional to coin angle: m ≈ 0.59 sin(θ)
- [x] Speed of light c = 1 site/step (leading edge, independent of mass)
- [x] Group velocity decreases with mass
- [x] ±E symmetry (particle-antiparticle)
- [x] Delta functions spread (genuine dispersion, not conveyor belt)

### 3D Tests (need larger lattice, radius ~150+)

**A. Dispersion relation**
- [ ] E² = m² + k_x² + k_y² + k_z² (isotropic massive Dirac)
- [ ] Speed of light is the same in all directions
- [ ] Mass gap depends on coin angle the same way as 1D

**B. Propagation**
- [ ] Leading edge of a delta function propagates at c in all directions (light cone)
- [ ] Leading edge speed is independent of coin angle (= mass)
- [ ] Group velocity of a wavepacket decreases with mass
- [ ] Wavepacket spreading rate matches the 3D Dirac prediction

**C. Symmetries**
- [ ] Isotropy: <x²> ≈ <y²> ≈ <z²> at all times (already confirmed to ~1%)
- [ ] ±E symmetry in the spectrum (particle-antiparticle)
- [ ] Spin degeneracy (double degeneracy of each energy level)
- [ ] CPT symmetry: check behavior under charge conjugation, parity, time reversal

**D. Density of states**
- [ ] DOS ~ E² near E=0 for massless case (3D Dirac)
- [ ] DOS has a gap for massive case, with van Hove singularity at E=m
- [ ] Compare numerically with the analytical 3D Dirac DOS

**E. Interference and coherence**
- [ ] Two-slit experiment: does interference pattern match Dirac prediction?
  (Connected to Świerczkowski unique path question)
- [ ] Wavepacket maintains coherence over long propagation
- [ ] Zitterbewegung (trembling motion) — characteristic Dirac oscillation
  between positive and negative energy components

**F. External fields (future)**
- [ ] Coupling to electromagnetic field via minimal substitution
- [ ] Klein paradox: transmission through potential barrier > mc²
- [ ] Hydrogen-like spectrum from Coulomb potential

**G. Continuum limit tests**
- [ ] Scale invariance: results at different lattice spacings collapse
  onto the same curve when expressed in physical units
- [ ] Compare wavepacket evolution at step 10 on lattice A with step 20
  on lattice B (twice as fine), rescaled appropriately

### What would be most convincing to a skeptic

The single most convincing test: **isotropic dispersion E² = m² + |k|²** on the 3D lattice, with the mass controlled by the coin angle. This directly demonstrates:
1. Lorentz invariance (isotropy + correct E-k relationship)
2. The role of the coin as mass
3. The correct relativistic energy-momentum relation

Second most convincing: **light cone propagation** — the leading edge of a localized wavepacket expands as a sphere at speed c, regardless of mass. This is visually striking and uniquely relativistic.

**Why:** These tests would demonstrate that the walk reproduces the kinematic structure of the Dirac equation (dispersion, symmetries, propagation) on a fundamentally discrete, aperiodic geometry.

**How to apply:** Implement these tests as we scale to larger lattices. Prioritize A and B (dispersion and propagation), which require radius ~150+. Tests C and D can be done on smaller lattices. Tests E, F, G are longer-term goals.
