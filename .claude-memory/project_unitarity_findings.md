---
name: Unitarity findings for shift operator
description: 3D walk now exactly unitary; chainPrepend index ordering bug was root cause; shift refactored into pure + overflow
type: project
---

Key findings from unitarity investigation (2026-03-29):

1. **3D walk is now exactly unitary**: norm = 1.000000 at every step until lattice capacity reached, matching 1D walk.

2. **Root cause**: `chainPrepend` in lattice.d decremented `rootIdx` before incrementing existing sites' `rIdx`. This made `exitDirForSite` return the wrong BC helix formula index (off by -1) when computing `bwdBlock` for the old first site and `fwdBlock` for the new prepended site. Fix: shift existing indices immediately after `rootIdx--`, before any ops computation.

3. **Shift refactored** into `pureShift` (pure linear algebra, returns overflow at chain ends) and `handleOverflow` (lattice mutation — extend or absorb). Legacy `applyShift` wraps both. This decouples the unitary operator from site creation/destruction.

4. **Closed chains** (Python test) are exactly unitary to machine precision (10⁻¹⁵) over 100+ steps — validates manifold approach.
