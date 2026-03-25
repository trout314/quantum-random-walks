/**
 * lattice.d — Tetrahedral lattice: chain-based site storage and generation.
 *
 * Sites live at the intersection of R and L helix chains. Each chain stores
 * an array of site IDs. A site knows which R-chain and L-chain it belongs to,
 * and its index along each. Chain links (next/prev) are implicit: the next
 * site on chain C is just C.siteIds[index + 1].
 *
 * No spatial hash is needed: sites are created by chain extension (guaranteed
 * unique by the no-loops property), and perpendicular chains share sites via
 * explicit parent references, not position lookup.
 */
module lattice;

import std.math : sqrt, exp, fabs;
import std.complex : Complex;
import geometry : Vec3, dot, norm, helixStep, reorth, initTet;

alias C = Complex!double;

/// BC helix face patterns.
immutable int[4] PAT_R = [1, 3, 0, 2];
immutable int[4] PAT_L = [0, 1, 2, 3];

/// Next face in the cyclic pattern after cur_face.
int nextFace(const int[4] pat, int curFace) {
    foreach (i; 0 .. 4)
        if (pat[i] == curFace) return pat[(i + 1) % 4];
    return pat[0];
}

/// Previous face in the cyclic pattern before cur_face.
int prevFace(const int[4] pat, int curFace) {
    foreach (i; 0 .. 4)
        if (pat[i] == curFace) return pat[(i + 3) % 4];
    return pat[0];
}

/// A helix chain: an ordered list of site IDs with root geometry.
struct Chain {
    bool isR;
    int rootSite;        /// site ID where this chain was spawned
    int[] siteIds;       /// site IDs in order along the chain
    int rootIdx;         /// index of rootSite within siteIds
}

/// Per-site data.
struct Site {
    Vec3 pos;
    Vec3[4] dirs;
    int rChain = -1, rIdx = -1;  /// R-chain membership
    int lChain = -1, lIdx = -1;  /// L-chain membership
    int rFace = -1, lFace = -1;  /// face index for each chain
}

/// Density grid for limiting site count per unit volume.
struct DensityGrid {
    int[] counts;
    double gridHalf;
    int gridN;
    int maxPerCell;

    static DensityGrid create(double halfExtent, int maxPer) {
        DensityGrid g;
        g.gridHalf = halfExtent;
        g.gridN = cast(int)(2.0 * halfExtent / 1.0) + 1;
        if (g.gridN > 500) g.gridN = 500;
        g.counts = new int[g.gridN * g.gridN * g.gridN];
        g.counts[] = 0;
        g.maxPerCell = maxPer;
        return g;
    }

    int idx(Vec3 pos) const {
        int gx = cast(int)((pos.x + gridHalf) / 1.0);
        int gy = cast(int)((pos.y + gridHalf) / 1.0);
        int gz = cast(int)((pos.z + gridHalf) / 1.0);
        if (gx < 0) gx = 0; if (gx >= gridN) gx = gridN - 1;
        if (gy < 0) gy = 0; if (gy >= gridN) gy = gridN - 1;
        if (gz < 0) gz = 0; if (gz >= gridN) gz = gridN - 1;
        return gx * gridN * gridN + gy * gridN + gz;
    }

    bool isFull(Vec3 pos) const {
        return counts[idx(pos)] >= maxPerCell;
    }

    void increment(Vec3 pos) {
        counts[idx(pos)]++;
    }
}

/// The lattice.
struct Lattice {
    Site[] sites;
    C[] psi;
    C[] tmp;
    Chain[] chains;
    int nsites;
    int maxSites;

    int[] freeList;
    int freeCount;

    static Lattice create(int maxSites) {
        Lattice lat;
        lat.maxSites = maxSites;
        lat.sites = new Site[maxSites];
        lat.psi = new C[4 * maxSites];
        lat.psi[] = C(0, 0);
        lat.tmp = new C[4 * maxSites];
        lat.tmp[] = C(0, 0);
        lat.nsites = 0;
        lat.freeList = new int[maxSites];
        lat.freeCount = 0;
        return lat;
    }

    int allocSite(Vec3 pos, Vec3[4] dirs) {
        int id;
        if (freeCount > 0)
            id = freeList[--freeCount];
        else {
            assert(nsites < maxSites, "Too many sites");
            id = nsites++;
        }
        sites[id] = Site(pos, dirs);
        psi[4*id .. 4*id+4] = C(0, 0);
        return id;
    }

    void removeSite(int id) {
        psi[4*id .. 4*id+4] = C(0, 0);
        sites[id] = Site.init;
        freeList[freeCount++] = id;
    }

    void swapBuffers() {
        auto swap = psi;
        psi = tmp;
        tmp = swap;
    }

    /// Next/prev site on a chain (implicit from chain's siteIds array).
    int chainNext(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx < 0) return -1;
        int nextIdx = idx + 1;
        if (nextIdx >= chains[chainId].siteIds.length) return -1;
        return chains[chainId].siteIds[nextIdx];
    }

    int chainPrev(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx <= 0) return -1;
        return chains[chainId].siteIds[idx - 1];
    }

    int chainFace(int siteId, bool isR) const {
        return isR ? sites[siteId].rFace : sites[siteId].lFace;
    }

    void setChainFace(int siteId, bool isR, int face) {
        if (isR) sites[siteId].rFace = face; else sites[siteId].lFace = face;
    }

    /// Append a site to the forward end of a chain.
    void chainAppend(int chainId, int siteId) {
        chains[chainId].siteIds ~= siteId;
        int idx = cast(int) chains[chainId].siteIds.length - 1;
        if (chains[chainId].isR) {
            sites[siteId].rChain = chainId;
            sites[siteId].rIdx = idx;
        } else {
            sites[siteId].lChain = chainId;
            sites[siteId].lIdx = idx;
        }
    }

    /// Prepend a site to the backward end of a chain (shifts all indices).
    void chainPrepend(int chainId, int siteId) {
        // Shift existing indices
        bool isR = chains[chainId].isR;
        foreach (existingId; chains[chainId].siteIds) {
            if (isR) sites[existingId].rIdx++;
            else     sites[existingId].lIdx++;
        }
        chains[chainId].rootIdx++;

        // Insert at front
        chains[chainId].siteIds = [siteId] ~ chains[chainId].siteIds;
        if (isR) {
            sites[siteId].rChain = chainId;
            sites[siteId].rIdx = 0;
        } else {
            sites[siteId].lChain = chainId;
            sites[siteId].lIdx = 0;
        }
    }
}

/// Seed queue entry.
private struct ChainSeed { int siteId; bool isR; }

/// Generate all sites using chain-first approach.
int generateSites(ref Lattice lat, double sigma, double seedThresh,
                  ref DensityGrid grid) {
    auto d0 = initTet();
    int origin = lat.allocSite(Vec3(0, 0, 0), d0);
    grid.increment(Vec3(0, 0, 0));

    auto queue = new ChainSeed[2 * lat.maxSites + 2];
    int qHead = 0, qTail = 0;
    queue[qTail++] = ChainSeed(origin, true);
    queue[qTail++] = ChainSeed(origin, false);

    int nChains = 0;

    while (qHead < qTail) {
        auto seed = queue[qHead++];
        int rootSite = seed.siteId;
        bool isR = seed.isR;

        if (lat.chainFace(rootSite, isR) >= 0) continue;

        const int[4] pat = isR ? PAT_R : PAT_L;
        lat.setChainFace(rootSite, isR, pat[0]);

        // Create chain with root as only member
        int chainId = cast(int) lat.chains.length;
        lat.chains ~= Chain(isR, rootSite, [rootSite], 0);
        if (isR) {
            lat.sites[rootSite].rChain = chainId;
            lat.sites[rootSite].rIdx = 0;
        } else {
            lat.sites[rootSite].lChain = chainId;
            lat.sites[rootSite].lIdx = 0;
        }

        // Extend forward
        int fwd = extendDir(lat, chainId, true, sigma, seedThresh, grid, queue, qTail);
        // Extend backward
        int bwd = extendDir(lat, chainId, false, sigma, seedThresh, grid, queue, qTail);

        if (fwd + bwd > 0) nChains++;
    }

    return nChains;
}

/// Extend a chain in one direction, creating new sites.
private int extendDir(ref Lattice lat, int chainId, bool forward,
                      double sigma, double seedThresh,
                      ref DensityGrid grid,
                      ChainSeed[] queue, ref int qTail) {
    auto ch = &lat.chains[chainId];
    bool isR = ch.isR;
    const int[4] pat = isR ? PAT_R : PAT_L;

    // Start from the appropriate end of the chain
    int endSite = forward ? ch.siteIds[$ - 1] : ch.siteIds[0];
    int curFace = lat.chainFace(endSite, isR);
    Vec3 p = lat.sites[endSite].pos;
    Vec3[4] d = lat.sites[endSite].dirs;

    int created = 0;

    for (int step = 0; step < lat.maxSites; step++) {
        int stepFace = forward ? curFace : prevFace(pat, curFace);
        helixStep(p, d, stepFace);
        if ((step + 1) % 8 == 0) reorth(d);

        if (exp(-dot(p, p) / (2 * sigma * sigma)) < seedThresh) break;
        if (grid.isFull(p)) break;

        int nbFace = forward ? nextFace(pat, curFace) : stepFace;

        Vec3[4] dd = d;
        reorth(dd);
        int nb = lat.allocSite(p, dd);
        lat.setChainFace(nb, isR, nbFace);
        grid.increment(p);

        if (forward)
            lat.chainAppend(chainId, nb);
        else
            lat.chainPrepend(chainId, nb);

        queue[qTail++] = ChainSeed(nb, !isR);

        curFace = nbFace;
        created++;

        // Re-fetch chain pointer (appending may have reallocated)
        ch = &lat.chains[chainId];
    }

    return created;
}

// ---- D unit tests ----

unittest {
    foreach (f; 0 .. 4) {
        assert(prevFace(PAT_R, nextFace(PAT_R, f)) == f);
        assert(nextFace(PAT_R, prevFace(PAT_R, f)) == f);
    }
}

unittest {
    auto lat = Lattice.create(100000);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    int nChains = generateSites(lat, sigma, 1e-4, grid);

    assert(lat.nsites > 1);
    assert(nChains > 0);
    assert(lat.chains.length > 0);

    assert(lat.chainFace(0, true) >= 0);
    assert(lat.chainFace(0, false) >= 0);

    // Chain next/prev consistency
    foreach (s; 0 .. lat.nsites) {
        int nxt = lat.chainNext(s, true);
        if (nxt >= 0) {
            int prv = lat.chainPrev(nxt, true);
            assert(prv == s, "R-chain next/prev mismatch");
        }
        nxt = lat.chainNext(s, false);
        if (nxt >= 0) {
            int prv = lat.chainPrev(nxt, false);
            assert(prv == s, "L-chain next/prev mismatch");
        }
    }
}

unittest {
    auto lat = Lattice.create(100);
    auto d = initTet();
    int id0 = lat.allocSite(Vec3(0, 0, 0), d);
    int id1 = lat.allocSite(Vec3(1, 0, 0), d);
    assert(lat.nsites == 2);

    lat.removeSite(id0);
    assert(lat.freeCount == 1);

    int id2 = lat.allocSite(Vec3(2, 0, 0), d);
    assert(id2 == id0);
    assert(lat.freeCount == 0);
}
