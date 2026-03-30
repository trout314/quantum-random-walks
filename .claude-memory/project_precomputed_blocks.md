---
name: Precomputed shift blocks optimization plan
description: Next optimization target: precompute fwdBlock/bwdBlock per chain link, store in deque-style ChainBuffer
type: project
---

The shift operator currently recomputes τ, P±, frame transport, and the shift block matrix (2 × 4×4 mat-mul) for every site every step. This is ~1000 FLOPs per site per direction vs ~200 cycles for a cache miss.

**Plan: Precompute per-chain-link shift blocks**

Each link between sites i and i+1 on a chain needs:
- fwdBlock = frameTransport(τ_i, τ_{i+1}) · P+(τ_i) — applied to ψ(i), result to i+1
- bwdBlock = frameTransport(τ_{i+1}, τ_i) · P-(τ_{i+1}) — applied to ψ(i+1), result to i

Storage: ~17K chains × ~8 links × 256 bytes per block × 2 = ~70 MB total. Negligible.

**Issue: Chain extension/pruning during walk**

Chains grow at the wavefront and shrink from pruning. Need a deque-style buffer (ChainBuffer) with room on both ends:
- pushBack/pushFront: O(1) amortized
- popBack/popFront: O(1)
- Reallocation only when buffer exhausted, with centering

**Attempted and reverted:** Direct `~=` on D dynamic arrays triggered excessive GC allocation at the wavefront (~140K extensions per step), making it slower than computing on-the-fly.

**Alternative considered:** Compute on-the-fly only at chain ends (wavefront), use precomputed blocks for interior. This gave 4× shift speedup but the fallback path had a bug (sites extended in previous steps had null blocks). Fix: either ChainBuffer for proper deque, or compute-and-store when extending.

**How to apply:** Implement ChainBuffer!T as a simple deque struct. Replace Chain's int[] siteIds with ChainBuffer!int. Add ChainBuffer!Mat4 for fwdBlocks/bwdBlocks. Update chainAppend/chainPrepend and the pruning unlink to use pushBack/pushFront/popBack/popFront.
