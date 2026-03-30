---
name: Chain-end pruning plan
description: Plan for periodic memory reclamation by pruning low-amplitude chain-end sites with free-list ID reuse
type: project
---

Periodic pruning of low-amplitude chain-end sites to keep site budget focused on where probability lives.

**Approach: Free list with ID reuse (option a)**

1. Every K steps, scan all sites for chain ends (r_next=-1 or r_prev=-1, and similarly for L-chains)
2. A chain-end site X_n is prunable when BOTH:
   - |psi(X_n)|² < threshold
   - Flow of probability from X_{n-1} to X_n < threshold (i.e. the P+ projection of psi at X_{n-1} along that chain direction is also small)
3. To prune: unlink X_n (set neighbor's next/prev to -1), remove from hash table (tombstone), zero its psi, push ID onto free list
4. site_insert pops from free list before incrementing nsites
5. Iterate: after pruning, newly exposed chain ends may also be prunable — repeat scan until no more prunable sites
6. Hash table uses tombstone markers for removed entries (already open-addressing)

**Why:** The wavefront expands while interior probability thins out. Without pruning, old tail sites consume site budget that could be used for wavefront expansion. This extends how many time steps we can run within MAX_SITES.

**How to apply:** Implement in walk_adaptive.c. The dual check (amplitude at X_n AND flow from X_{n-1}) prevents premature pruning at the active wavefront. By the no-loops property, X_n can only receive amplitude from X_{n-1} on its chain, so pruning is always safe when both conditions are met.

**Key safety property:** No site other than X_{n-1} can send amplitude to X_n (no-loops), so pruned sites cannot miss incoming probability from unexpected sources.
