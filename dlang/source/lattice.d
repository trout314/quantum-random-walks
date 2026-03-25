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
import geometry : Vec3, dot, norm, helixStep, reorth, initTet;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul;

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

/// Deque backed by a flat array. Active elements are buf[lo .. hi].
/// Grows by doubling and centering when either end is exhausted.
struct Deque(T) {
    T[] buf;
    int lo, hi;  /// active range [lo, hi)

    @property int length() const { return hi - lo; }

    ref inout(T) opIndex(int i) inout { return buf[lo + i]; }

    void pushBack(T val) {
        if (hi >= buf.length) expand();
        buf[hi++] = val;
    }

    void pushFront(T val) {
        if (lo <= 0) expand();
        buf[--lo] = val;
    }

    void popBack() { hi--; }
    void popFront() { lo++; }

    @property T back() const { return buf[hi - 1]; }
    @property T front() const { return buf[lo]; }

    /// Ensure at least minCap total buffer slots.
    void reserve(int minCap) {
        if (buf.length >= minCap) return;
        int n = length;
        auto newBuf = new T[minCap];
        int newLo = (minCap - n) / 2;
        if (n > 0)
            newBuf[newLo .. newLo + n] = buf[lo .. hi];
        buf = newBuf;
        lo = newLo;
        hi = newLo + n;
    }

    /// Double the buffer, centering the active range.
    private void expand() {
        int n = length;
        int newCap = (buf.length < 16) ? 32 : cast(int)(buf.length * 2);
        auto newBuf = new T[newCap];
        int newLo = (newCap - n) / 2;
        if (n > 0)
            newBuf[newLo .. newLo + n] = buf[lo .. hi];
        buf = newBuf;
        lo = newLo;
        hi = newLo + n;
    }
}

/// A helix chain with precomputed shift blocks.
struct Chain {
    bool isR;
    int rootSite;
    Deque!int siteIds;
    int rootIdx;

    /// Shift blocks for each link. fwdBlocks[i] applied to ψ(siteIds[i]),
    /// result added to tmp(siteIds[i+1]). Length = siteIds.length - 1.
    Deque!Mat4 fwdBlocks;
    Deque!Mat4 bwdBlocks;
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
    double[] psiRe, psiIm;   /// spinor: 4 components per site, split real/imag
    double[] tmpRe, tmpIm;    /// scratch buffer for shift
    Chain[] chains;
    int nsites;
    int maxSites;

    int[] freeList;
    int freeCount;

    static Lattice create(int maxSites) {
        Lattice lat;
        lat.maxSites = maxSites;
        lat.sites = new Site[maxSites];
        lat.psiRe = new double[4 * maxSites];
        lat.psiIm = new double[4 * maxSites];
        lat.psiRe[] = 0;
        lat.psiIm[] = 0;
        lat.tmpRe = new double[4 * maxSites];
        lat.tmpIm = new double[4 * maxSites];
        lat.tmpRe[] = 0;
        lat.tmpIm[] = 0;
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
        psiRe[4*id .. 4*id+4] = 0;
        psiIm[4*id .. 4*id+4] = 0;
        return id;
    }

    void removeSite(int id) {
        psiRe[4*id .. 4*id+4] = 0;
        psiIm[4*id .. 4*id+4] = 0;
        sites[id] = Site.init;
        freeList[freeCount++] = id;
    }

    void swapBuffers() {
        auto swapRe = psiRe;
        auto swapIm = psiIm;
        psiRe = tmpRe;
        psiIm = tmpIm;
        tmpRe = swapRe;
        tmpIm = swapIm;
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

    /// Get the precomputed forward block for the link from siteId to its successor.
    const(Mat4)* fwdBlock(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx < 0) return null;
        if (idx >= chains[chainId].fwdBlocks.length) return null;
        return &chains[chainId].fwdBlocks[idx];
    }

    /// Get the precomputed backward block for the link from siteId to its predecessor.
    const(Mat4)* bwdBlock(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx <= 0) return null;
        if (idx - 1 >= chains[chainId].bwdBlocks.length) return null;
        return &chains[chainId].bwdBlocks[idx - 1];
    }

    /// Append a site to the forward end of a chain with shift block.
    void chainAppend(int chainId, int siteId) {
        auto ch = &chains[chainId];

        // Compute shift block for the new link before appending
        if (ch.siteIds.length > 0) {
            int prev = ch.siteIds[ch.siteIds.length - 1];
            int facePrev = ch.isR ? sites[prev].rFace : sites[prev].lFace;
            int faceNew = ch.isR ? sites[siteId].rFace : sites[siteId].lFace;
            Mat4 tauPrev = makeTau(sites[prev].dirs[facePrev]);
            Mat4 tauNew = makeTau(sites[siteId].dirs[faceNew]);
            ch.fwdBlocks.pushBack(mul(frameTransport(tauPrev, tauNew), projPlus(tauPrev)));
            ch.bwdBlocks.pushBack(mul(frameTransport(tauNew, tauPrev), projMinus(tauNew)));
        }

        ch.siteIds.pushBack(siteId);
        int idx = ch.siteIds.length - 1;
        if (ch.isR) {
            sites[siteId].rChain = chainId;
            sites[siteId].rIdx = idx;
        } else {
            sites[siteId].lChain = chainId;
            sites[siteId].lIdx = idx;
        }
    }

    /// Prepend a site to the backward end of a chain with shift block.
    void chainPrepend(int chainId, int siteId) {
        auto ch = &chains[chainId];
        bool isR = ch.isR;

        // Compute shift block for the new link before prepending
        if (ch.siteIds.length > 0) {
            int next = ch.siteIds[0];
            int faceNext = isR ? sites[next].rFace : sites[next].lFace;
            int faceNew = isR ? sites[siteId].rFace : sites[siteId].lFace;
            Mat4 tauNew = makeTau(sites[siteId].dirs[faceNew]);
            Mat4 tauNext = makeTau(sites[next].dirs[faceNext]);
            ch.fwdBlocks.pushFront(mul(frameTransport(tauNew, tauNext), projPlus(tauNew)));
            ch.bwdBlocks.pushFront(mul(frameTransport(tauNext, tauNew), projMinus(tauNext)));
        }

        // Shift existing indices
        for (int i = 0; i < ch.siteIds.length; i++) {
            int existingId = ch.siteIds[i];
            if (isR) sites[existingId].rIdx++;
            else     sites[existingId].lIdx++;
        }
        ch.rootIdx++;

        ch.siteIds.pushFront(siteId);
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
        Chain newChain;
        newChain.isR = isR;
        newChain.rootSite = rootSite;
        newChain.siteIds.pushBack(rootSite);
        newChain.rootIdx = 0;
        lat.chains ~= newChain;
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

    // Reserve extra capacity in all chain deques for walk-time growth.
    // Each chain may grow by ~nSteps sites on each end.
    foreach (ref ch; lat.chains) {
        int extra = ch.siteIds.length + 200;  // room for ~200 steps of growth
        ch.siteIds.reserve(extra);
        ch.fwdBlocks.reserve(extra);
        ch.bwdBlocks.reserve(extra);
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
    int endSite = forward ? ch.siteIds[ch.siteIds.length - 1] : ch.siteIds[0];
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
