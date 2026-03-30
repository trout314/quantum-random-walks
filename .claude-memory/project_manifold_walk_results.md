---
name: Manifold walk results and structure
description: 3D walk on triangulated manifold: A4 fiber bundle, exact unitarity, spectrum, wavepacket tests, interference via coarse-graining
type: project
---

## Manifold Walk Implementation (2026-03-29)

**Structure**: The 3D lattice on a closed manifold is a 12-fold cover (A₄ fiber bundle) of the triangulation. Each tet has exactly 12 sites, one per even permutation of its 4 vertices. Each site lies on exactly one R-chain and one L-chain — 12 R-passages and 12 L-passages per tet, covering all 12 pairs of perpendicular spiral axes.

**Unitarity**: Exactly unitary on closed chains. Norm = 1.000000000 for 200+ steps, max |norm-1| ~ 10⁻¹³. Verified on manifolds up to 10K tets (120K sites).

**Physics of interference**: Waves along different spirals pass through each other without interacting (12 independent channels per tet). Interference happens at observation time via coarse-graining (coherent sum with frame transport). This is correct — matches continuum Dirac where plane waves pass through each other.

**Spreading**: PR saturates at ~0.80×N (universal across triangulations). Tet probability std/mean ≈ 15%. Negative correlation between vertex degree and tet probability. Return probability drops to ~10⁻⁵ with coherent revivals.

**Spectrum** (equilibrated_200, 5328D, φ=0.05): All |λ|=1 exactly. Nearly flat density of states (no gap, no band structure). Gap statistics std/mean ≈ 0.43 (close to CUE random matrix value ~0.52). Walk is quantum-chaotic/ergodic on this small manifold.

**Wavepacket kick tests**: Momentum kicks using 3D helix coordinates don't work — positions are locally valid but globally inconsistent across different chains, making e^{ikx} incoherent as a manifold momentum. Frame-transported balanced IC (1,0,i,0)/√2 was factored out as shared initWavepacket. Need larger manifold (10K+ tets, diameter 95+) for Dirac-like dispersion to emerge. Mass m ≈ 0.044 at φ=0.05 means need k ~ 0.01-0.1 and σ ~ 3-5 for visible effects.

**Chain tracing**: Done in D for performance. BFS discovers all R/L chains from a single seed. Instant for 10K tets.

**Key parameters**: φ=0.05 gives mass m = 0.878×φ ≈ 0.044. Compton wavelength 1/m ≈ 23 lattice steps. Need manifold diameter >> 23 for clean wavepacket tests.

**Files**:
- `dlang/source/manifold_interop.d` — D interface: chain tracing, lattice build, walk step, observables, IC
- `dlang/source/operators.d` — shared `initWavepacket` (BFS frame transport + Gaussian + kick), `pureShift` with closed-chain support
- `dlang/source/lattice.d` — `Chain.isClosed`, `closeChain()`, `chainPrepend` fix (index ordering bug)
- `scripts/manifold_walk_driver.py` — Python driver for spreading tests (PR, return prob, tet histogram)
- `scripts/manifold_kick_test.py` — Wavepacket kick tests (3D coordinates)
- `scripts/manifold_spectrum.py` — Walk operator eigenvalue analysis

**Open questions**:
- How to define coherent momentum on the manifold (spectral embedding? Laplacian eigenvectors?)
- Is the flat DOS a small-manifold artifact or universal?
- Can we see Dirac dispersion on a larger manifold with enough room for wavepacket propagation?
