---
name: feedback_no_loops
description: Always respect the no-loops (Świerczkowski) property when reasoning about the tetrahedral walk
type: feedback
---

When reasoning about the walk, never propose explanations that violate the no-loops property: walks on the tetrahedral lattice never revisit a vertex (Świerczkowski). Key consequences:

1. Each site has exactly one R-chain and one L-chain through it.
2. Two different chains of the same chirality cannot pass through the same site.
3. There is NO interference in the walk — for a given chain (R or L), each site receives P+ from its unique predecessor and P- from its unique successor. These are the only two sources. There is no third contributor that could cause "constructive interference." Do not invoke interference as an explanation for norm issues.
4. Do not invoke "duplicate contributions from multiple chains" as an explanation for bugs.

**Why:** The user has corrected these reasoning errors multiple times. The no-loops/no-interference property is fundamental to the walk's structure and must constrain all analysis.

**How to apply:** Before attributing any norm/amplification issue to interference, overlapping chains, or duplicate entries, verify whether the proposed scenario violates the unique-path property. If it does, discard the explanation and look for the real cause (e.g., mutating data structures mid-computation, incorrect face/link assignments).
