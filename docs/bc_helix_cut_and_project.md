# The BC Helix as a 1D Quasicrystal via Cut-and-Project

The Boerdijk-Coxeter (BC) helix — a chain of regular tetrahedra joined
face-to-face — has no translational symmetry. Its twist angle per tetrahedron,
θ = arccos(−2/3) ≈ 131.81°, is incommensurate with 2π, so the helix never
repeats. This makes it a 1D quasicrystal, and like other quasicrystals it
admits a description as an irrational-angle slice through a higher-dimensional
periodic structure.

## The 600-cell makes it periodic

The {3,3,5} polytope (600-cell) in 4D tiles the 3-sphere S³ with 600 regular
tetrahedra. Its 120 vertices all lie on S³ at radius φ (golden ratio) when
edges have unit length. The 600-cell decomposes into **20 BC helix rings**,
each containing 30 tetrahedra, forming a discrete Hopf fibration. On S³ each
ring is a **(30, 11) torus knot** on a Clifford torus — it winds 11 times
around one generating circle for every 30 tetrahedra. On S³ the helix is
**perfectly periodic** with period 30.

## The triangular lattice and strip projection

Sadoc & Rivier (1999) flatten the Clifford torus to obtain a **2D triangular
lattice**. The BC helix corresponds to a strip through this lattice along a
direction with **irrational slope**. The winding number is

    tetrahedra per turn = 1 + √3 ≈ 2.732

which has continued fraction expansion [2, 1, 2, 1, 2, …]. The best rational
approximant on S³ is 30/11.

Projecting this strip onto 1D gives the aperiodic vertex sequence — exactly
the standard strip/cut-and-project method for constructing a 1D quasicrystal
from a 2D periodic lattice.

### Lattice vectors

On the triangular lattice with basis vectors a₁, a₂, Sadoc & Rivier identify
the "multiple cell" vectors:

    b₁ = 3a₁ + 2a₂   (winding direction)
    b₂ = −3a₁ + 8a₂   (axial direction)

with b₁ · b₂ = −2 (nearly orthogonal). Rolling the strip along b₂ into a
cylinder gives the BC helix; closing along b₁ gives the torus in S³ where
the helix becomes periodic.

## The decurving interpretation

The conceptual picture is one of **geometric frustration** (Sadoc & Mosseri):

1. **On S³**: the 600-cell is a perfect periodic tetrahedral crystal.
2. **On the Clifford torus**: the BC helix traces a (30,11) torus knot on a
   triangular lattice — periodic.
3. **Unroll to flat space**: the curvature that closed the 30/11 winding is
   lost, arccos(−2/3) per tetrahedron is incommensurate with 2π, and the
   helix becomes aperiodic.

Regular tetrahedra tile curved S³ perfectly but cannot tile flat R³. The
curvature mismatch manifests as the irrational twist.

## Cut-and-project dictionary

| Quasicrystal concept          | BC helix realization                                           |
|-------------------------------|----------------------------------------------------------------|
| Higher-dim periodic lattice   | Triangular lattice on Clifford torus in S³ (vertices of 600-cell) |
| Irrational slope / subspace   | Helix axis direction, slope = 1+√3 in lattice coords          |
| Strip / acceptance window     | Strip of triangular lattice along helix direction              |
| Physical (parallel) space     | 1D projection along helix axis                                |
| Internal (perpendicular) space| Transverse direction on torus; curling it restores periodicity |

## The "curled up dimension" duality

Fang & Irwin (2021) make explicit that for any 1D quasicrystal from a 2D
lattice, **curling the perpendicular space into a loop** converts the
aperiodic structure into a periodic one. For the BC helix this is precisely
the reverse of Sadoc-Rivier: going from the flat strip (aperiodic) to the
torus on S³ (periodic with period 30).

## Periodic approximants

Talis & Kucherinenko (2023) show that the irrational 30/11 screw axis of the
BC helix admits **rational periodic approximants** — notably 11/4 and 8/3 —
which correspond to helices of slightly deformed tetrahedra that do tile flat
R³. These approximants appear in real crystal structures (α-Mn, β-Mn).

## E8 and H4 connections

The 600-cell has symmetry group H₄ (order 14400). The E₈ root lattice in 8D
projects onto H₄ representations in 4D, yielding two concentric 600-cells
scaled by the golden ratio. BC helices appear as substructures within this
E₈ → H₄ projection.

## References

1. **Sadoc & Rivier**, "Boerdijk-Coxeter helix and biological helices,"
   *Eur. Phys. J. B* 12, 309–318 (1999).
   https://link.springer.com/article/10.1007/s100510051009

2. **Sadoc & Mosseri**, *Geometrical Frustration*, Cambridge University Press
   (1999).

3. **Fang, Irwin et al.**, "The Curled Up Dimension in Quasicrystals,"
   *Crystals* 11(10), 1238 (2021).
   https://www.mdpi.com/2073-4352/11/10/1238

4. **Talis & Kucherinenko**, "Non-crystallographic helices in polymers and
   close-packed metallic crystals determined by the four-dimensional
   counterpart of the icosahedron," *Acta Cryst. B* 79, 537–546 (2023).
   https://journals.iucr.org/paper?yh5028

5. **Sadler, Fang, Kovacs, Irwin**, "Periodic modification of the
   Boerdijk-Coxeter helix (tetrahelix)," *Mathematics* 7(10), 1001 (2019).
   https://arxiv.org/abs/1302.1174
