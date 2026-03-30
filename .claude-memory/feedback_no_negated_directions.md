---
name: feedback_no_negated_directions
description: Walker cannot step in -e_a directions; only 4 tetrahedral directions available per site, and they change from site to site
type: feedback
---

Do NOT assume that -e_a is an available step direction. Each site has exactly 4 tetrahedral directions, and -e_a is not among them. After stepping from site x via direction e_a, the directions at x + e_a are a DIFFERENT set of 4 tetrahedral directions (not the negation of the original set).

**Why:** The regular tetrahedron has no antipodal pairs — no vertex is opposite another. This is a fundamental geometric fact. The problem of determining the directions at each site (and ensuring global consistency of the resulting lattice) is one of the key open problems.

**How to apply:** When reasoning about the lattice structure, treat the direction sets at each site as unknowns to be determined. Don't assume bipartite/diamond structure or that -e_a is ever available.
