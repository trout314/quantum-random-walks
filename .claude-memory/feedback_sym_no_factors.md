---
name: Sym mode treats images as independent sites
description: In symmetric mode, all 12 image copies are full independent sites — no factors of 12 anywhere in thresholds, density limits, or probability accounting
type: feedback
---

Treat all A4 image sites as independent, identical sites. The only thing symmetry changes is that we store 1/12 of them and reconstruct the rest. No factors of 12 in thresholds, density limits, cutoffs, or probability flow. The per-site amplitudes, normalization, and creation decisions should be identical to what they'd be if all 12 copies were explicitly stored.

**Why:** Repeatedly introduced factors of 12 in thresh2, normalization, absorbed probability, etc., creating cascading confusion. The symmetric mode is purely a storage optimization, not a physics change.

**How to apply:** When implementing or modifying sym mode, ask: "would this line be different if all 12 copies were explicitly stored as real sites?" If no, don't add a factor of 12.
