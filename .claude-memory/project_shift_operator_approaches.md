---
name: project_shift_operator_approaches
description: Three candidate approaches for constructing unitary shift operators on the tetrahedral lattice
type: project
---

Candidate approaches for making the walkers walk:

1. **Split-step with position-dependent coins.** W = S_4 · S_3 · S_2 · S_1, where S_a uses the local τ_a at the walker's current site. Each S_a is unitary (conditional translation on ±1 eigenspaces). Product is unitary. Risk: unwanted BCH corrections since each sub-step changes the τ operators seen by subsequent sub-steps.

2. **Paired-site swaps.** Direction a pairs up sites (x, x+e_a). S_a acts as an involution swapping coin-space components between paired sites. The anti-conjugation relation (e_s·α) τ_a (e_s·α) = -τ'_a connects eigenspaces across a step and could guide the swap definition. Unitary by construction.

3. **Single-direction staggered walk.** Each time step uses only ONE direction (cycling a=0,1,2,3). Walker moves forward or stays based on τ_a. Simpler but 4 steps per effective step, and cycling breaks tetrahedral symmetry.

**Why:** These are the main strategies identified so far for the core open problem (unitarity of the walk operator on the tetrahedral lattice).

**How to apply:** When working on the shift operator construction, evaluate each approach. The user may also have their own idea to explore.
