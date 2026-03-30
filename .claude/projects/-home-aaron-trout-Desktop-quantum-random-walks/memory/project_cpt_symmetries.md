---
name: CPT symmetry results
description: Full CPT analysis: AZ class CII at φ=0; standard C/P/T vs AZ operators; CPT not restored in continuum limit
type: project
---

CPT symmetry analysis of the 1D walk W = V·S (Tests D/D2/D3 in scripts/).

## AZ classification (φ=0): class CII

- Chiral Γ = γ⁰γ⁵ (anticommutes with every τ_n): ΓWΓ† = W†
- TRS T_AZ = γ⁰γ²·K with T²=-1 (Kramers): TW*T† = W†
- PHS C_AZ = γ²γ⁵·K with C²=-1: CW*C† = W
- Eigenvalues 2-fold degenerate (Kramers), ±E paired (chiral), |E| 4-fold
- 1D topological invariant: 2ℤ

## Factor-wise symmetry (φ≠0)

All three AZ symmetries broken as exact walk symmetries, but:
- γ⁰γ⁵ maps S→S† and V→V† individually
- ΓWΓ† = V†S† ≠ S†V† = W† (ordering issue since [V,S]≠0)
- ±E spectral pairing PROTECTED: V†S† = (SV)† has eigvals = conj(eigvals(W))
- Kramers degeneracy BROKEN (eigenvalues non-degenerate at φ≠0)
- (-1)^n staggering exact at all φ: (-1)^n W (-1)^n = -W → E↔E+π

## Standard Dirac C/P/T vs AZ operators

Operator algebra:
- C_std (iγ²γ⁰) = -i · T_AZ (γ⁰γ²) — standard charge conjugation IS AZ TRS up to phase
- T_std (iγ¹γ³) is NOT a walk symmetry at any φ — wrong commutation with τ
- CT_std = -iγ⁵ ≠ γ⁰γ⁵ = Γ_AZ — differs by factor of γ⁰
- The walk's symmetry operators differ from standard Dirac because τ = (√7/4)β + (3/4)(d·α) mixes β and α

## P and combined symmetries

- P (γ⁰ ⊗ spatial reversal): BROKEN at all φ. BC helix has definite handedness.
  Parity is an inter-walk symmetry (R↔L helices), not intra-walk.
- CP, PT, CT, CPT: all broken. CP/PT/CPT inherit P breaking; CT_std ≠ chiral operator.

## CPT NOT restored in continuum limit

- CPT residual ~0.6 at all energy cutoffs (|E|<0.1 through |E|<π)
- No improvement with chain length (N=20 to N=160)
- Breaking is geometric (helix chirality), not a lattice artifact
- Chiral/TRS/PHS residuals at φ≠0 are ∝ φ (physical mass breaking), also scale-independent
- CPT restoration would require the 3D walk on a manifold with spatial inversion symmetry

**Why:** Constrains topology, edge modes, and identifies which continuum symmetries survive discretization.
**How to apply:** Reference CII class for topology. Standard Dirac T is NOT the walk's TRS — use γ⁰γ² instead. CPT requires 3D manifold with inversion symmetry.
