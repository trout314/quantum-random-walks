---
name: BC helix cut-and-project description
description: BC helix is a 1D quasicrystal via strip projection from 2D triangular lattice; periodic on S³ in 600-cell with period 30; numerically confirmed for quantum walk
type: project
---

The BC helix admits a cut-and-project description as a 1D quasicrystal:

- The 600-cell ({3,3,5} polytope) on S³ decomposes into 20 BC helix rings of 30 tetrahedra each (discrete Hopf fibration)
- On the Clifford torus in S³, each ring is a (30,11) torus knot on a 2D triangular lattice — perfectly periodic
- Unrolling/flattening the torus gives a strip in the triangular lattice; projecting along the irrational-slope helix axis (winding number 1+√3) yields the aperiodic BC helix
- The perpendicular (internal) space is the transverse direction on the torus; curling it into a loop restores periodicity

**Numerically confirmed:** The perpendicular coordinate φ = nθ mod 2π (where θ = arccos(-2/3)) is the correct internal-space variable. It equidistributes in [0, 2π), and the walk operator's Fourier spectrum in φ decays to noise after l = ±1. Quasi-Bloch eigenstates ψ(n) = e^{ikn} F(φ(n)) work, with F smooth and nearly constant in the bulk.

**Why:** The cut-and-project framework provides an analytic handle on the walk: the walk operator becomes a small banded matrix in harmonic space (l ↔ l±1 coupling), independent of chain length N.

**How to apply:** Use φ = nθ mod 2π as the perpendicular coordinate. The quasi-Bloch basis e^{ikn} e^{imnθ} e^{i2πrn/4} decomposes the walk into a k-parametrized operator of fixed size 16(2M+1), with M=1 already sufficient for the main dispersion. Rational approximants (30/11, 11/4, 8/3) give periodic chains amenable to exact Bloch analysis.

Key references: Sadoc & Rivier (1999), Sadoc & Mosseri (1999), Fang & Irwin (2021), Talis & Kucherinenko (2023). Full writeup in docs/bc_helix_cut_and_project.md.
