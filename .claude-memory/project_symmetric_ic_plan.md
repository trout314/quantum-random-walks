---
name: Tetrahedral-symmetric IC plan
description: Plan for reducing site count by 1/12 using A4-symmetric initial conditions and wedge seeding
type: project
---

Exploit tetrahedral (A4) symmetry of Gaussian IC to seed only 1/12 of sites.

**Approach: Wedge seeding with virtual density counting**

1. Define a fundamental domain (1/12 of solid angle, one wedge of the A4 rotation group)
2. During site generation, only create sites whose position falls in the wedge
3. For density counting (max_per_cell), compute all 12 A4 images of each created site and increment the density grid at each image position — this prevents over-density at wedge boundaries
4. Walk operator runs unchanged on the wedge sites — symmetry is preserved by the IC, no boundary wrapping or spinor rotation needed
5. Chains that would exit the wedge are simply truncated by the Gaussian cutoff or density limit (slight chain structure differences at boundaries are in the low-probability tail)

**Why:** At sigma=5, the seed is millions of sites. 12x reduction lets us either use smaller sigma with the same site budget, or run much longer at the same sigma.

**How to apply:** Add a command-line flag (e.g. `sym=1`) to walk_adaptive.c. The only code changes are in `generate_sites_chain_first` (wedge test + image density counting). The walk evolution code is untouched.

**Key simplification:** No domain-boundary wrapping needed. The 12 copies interpenetrate slightly near boundaries but this is in the probability tail and doesn't affect results.
