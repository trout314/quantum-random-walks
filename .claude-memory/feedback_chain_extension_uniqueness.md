---
name: Chain extension always creates unique sites
description: New sites at BC chain ends are guaranteed distinct; all paths to them go through the new step
type: feedback
---

When extending a BC helix chain by one step, the new site is *always* a genuinely new site — it cannot coincide with any existing site. Furthermore, the only walker path reaching that new site must go through the newly created step.

**Why:** This is a consequence of the no-loops property of BC helices (Świerczkowski). The user corrected an assumption that chain extension might encounter existing sites and need hash lookup/dedup.

**How to apply:** When implementing chain extension, skip the hash table lookup for newly created chain-end sites — they're guaranteed unique. This simplifies the adaptive extension logic and means we don't need position/direction data to check for collisions at chain ends.
