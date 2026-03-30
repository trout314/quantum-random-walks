---
name: Chain is a quasicrystal
description: Walker chain has no translational symmetry; standard Bloch inapplicable but quasi-Bloch via cut-and-project works
type: feedback
---

The set of walker sites along a chain is a quasicrystal:
- No translational symmetries
- No screw-translational symmetries
- No unit cell
- **Standard** Bloch's theorem does not apply
- **Quasi-Bloch decomposition works** via cut-and-project: ψ(n) = e^{ikn} F_{n mod 4}(nθ mod 2π) where θ = arccos(-2/3)

**Why:** The face pattern [1,3,0,2] repeats with period 4, but the direction vectors are reflected at each step. The reflections are irrational rotations, so the geometry never exactly repeats. However, the variation is quasiperiodic with perpendicular-space coordinate φ = nθ mod 2π, enabling a quasi-Bloch framework.

**How to apply:** Never assume exact periodicity. Do use the quasi-Bloch ansatz with the perpendicular coordinate φ = nθ mod 2π and sublattice index n mod 4. Eigenstates are plane waves modulated by smooth functions of φ (nearly constant in bulk, structured at band edges).
