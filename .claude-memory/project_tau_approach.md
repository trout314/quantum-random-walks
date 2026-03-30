---
name: project_tau_approach
description: Core approach uses τ operators (eigenvalues ±1, multiplicity 2) satisfying Σ τ_a D_a = Σ α_i ∂_i; open problem is unitarity of shift operators
type: project
---

The approach to the tetrahedral quantum walk uses operators τ_1,...,τ_4:
- Each τ_a is 4×4 Hermitian, unitarily equivalent to diag(-1,-1,1,1)
- Dirac correspondence: Σ_a τ_a D_a = α_1 ∂_x + α_2 ∂_y + α_3 ∂_z
- This rewrites the Dirac Hamiltonian in tetrahedral coordinates

**Why:** This allows applying the same techniques as the cubic lattice QRW (where each axis has a ±1 coin operator and a conditional shift).

**How to apply:** The unsolved problem is constructing a unitary time-evolution operator from the τ-based shift operators, because the 4 tetrahedral directions are non-orthogonal (e_a · e_b = -1/3) so per-direction shifts don't commute.
