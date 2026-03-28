# 1D Walk Scaling Rules and Parameter Constraints

The regime where the quantum walk on the BC helix reproduces the 1+1D
Dirac equation requires the following hierarchy:

    1/N  <<  φ_mix  <<  1  <<  σ  <<  N

## Mass mapping

The effective Dirac mass is **m = 0.878 × φ_mix**, not φ_mix itself.
This was predicted by the quasi-Bloch W(k=0) eigenvalue and confirmed
by density-fit optimization across multiple parameter sets.

## Required conditions

### σ >> 1  (spatial continuum limit)

The wavepacket must span many lattice sites so that discreteness
artifacts are suppressed. The dominant momentum content is k ~ 1/σ,
which stays in the continuum regime k << 1 when σ >> 1.

**Violation:** lattice-scale oscillations in the density.

### φ_mix · σ >> 1  (mass gap resolution)

Equivalently m·σ >> 1. The wavepacket spans many Compton wavelengths
λ_C = 1/m, so it resolves the mass gap in the dispersion relation.
Without this, the momentum spread Δk ~ 1/(2σ) exceeds m and the
packet behaves as if massless.

**Violation:** massless-like behavior, no Zitterbewegung splitting.

### k₀ · σ >> 1  (momentum kick resolution)

For a wavepacket with momentum kick k₀, the kick must be well-defined
relative to the natural momentum spread Δk ~ 1/(2σ). Otherwise the
kick is buried in the Gaussian width and has no dynamical effect.

**Violation:** kick invisible, wavepacket behaves as if k₀ = 0.

### φ_mix << 1  (Dirac / continuum limit)

The walk's dispersion relation E(k) matches √(k² + m²) only for small
k and small m. At large φ_mix, higher-order lattice corrections to the
dispersion become significant. Agreement degrades above φ_mix ~ 0.2.

**Violation:** dispersion deviates from Dirac, mass ratio drifts from 0.878.

### N >> σ + t  (boundary avoidance)

The chain must be long enough that the wavepacket doesn't reach the
boundary during evolution. The maximum propagation speed is 1 (the
lattice speed of light), so the wavepacket at time t extends to
roughly ±(σ + t) from center.

For periodic BCs, wrapping causes self-interference. For open BCs,
edges cause reflections.

**Violation:** wrapping artifacts, spurious interference, norm loss at edges.

### t >> m · σ²  (splitting time)

For the two-peak Zitterbewegung structure to develop, the evolution
time must exceed the separation timescale. The peak splitting speed
is v ~ Δk/m ~ 1/(2mσ), so peaks separate by ~2σ (clear resolution)
after t ~ 4mσ².

**Violation:** boring single-peak density with no interesting structure.

### φ_mix · N >> 1  (spectral resolution)

The eigenvalue spacing on a chain of N sites is ~2π/N. The mass gap m
must exceed this spacing for the gap to appear in the discrete spectrum.

**Violation:** mass gap unresolved in eigenvalue spectrum.

## Summary table

| Condition     | Meaning                      | Violation symptom              |
|---------------|------------------------------|--------------------------------|
| σ >> 1        | Continuum spatial resolution | Lattice oscillations           |
| φ·σ >> 1      | Mass gap resolved            | Massless behavior, no split    |
| k₀·σ >> 1    | Kick well-defined            | Kick invisible                 |
| φ << 1        | Dirac regime                 | Lattice dispersion corrections |
| N >> σ + t    | No boundary effects          | Wrapping, reflections          |
| t >> mσ²     | Structure develops           | Single-peak, no Zitterbewegung |
| φN >> 1       | Gap in spectrum              | Mass gap unresolved            |

## Verified sweet spot

Best agreement (3.6% RMS relative error) achieved at:

    φ_mix = 0.10,  σ = 20,  t = 400,  N = 1200

Check:
- σ = 20 >> 1                          ✓
- φσ = 2.0 >> 1                        ✓
- φ = 0.10 << 1                        ✓
- N = 1200 >> σ + t = 420              ✓
- t = 400 >> mσ² = 0.088 × 400 = 35   ✓
- φN = 120 >> 1                        ✓

IC: P±-symmetric frame-transported, with Dirac mass m = 0.0877 = 0.878 × φ_mix.
