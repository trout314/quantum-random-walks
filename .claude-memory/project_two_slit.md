---
name: project_two_slit
description: Open question about unique walker paths, two-slit interference, and whether smearing restores standard behavior
type: project
---

Świerczkowski's 1958 theorem (no closed chain of face-stacked regular tetrahedra) implies that between any two walker locations there is exactly ONE walker path connecting them.

**The problem:** In a two-slit experiment, a source X and detector Y are connected by one unique path that goes through either slit A or slit B, not both. Opening/closing the other slit doesn't create a second path. So naively: no path interference, no interference pattern. Internal spinor interference along a single path can't encode spatial information about the other slit.

**Possible resolution:** Smearing out the walker wavefunction over multiple lattice sites could restore interference. If the "source" and "detector" are regions (not single sites), many pairs of sites contribute to the amplitude, and different pairs may route through different slits.

**Tests to run:**
1. Point source, point detector, two slits: does interference appear? (expect: no)
2. Smeared source/detector: does interference emerge? At what smearing scale?
3. Compare interference pattern with continuum Dirac prediction
4. How does visibility depend on smearing scale vs slit separation?

**Why:** If smearing restores standard interference, the Dirac equation emerges as an effective description at scales larger than the lattice spacing, with the unique-path property invisible at long distances. If it doesn't, the walk fundamentally differs from Dirac in an observable way.

**How to apply:** Implement after achieving a large enough 3D lattice for the basic dispersion tests.
