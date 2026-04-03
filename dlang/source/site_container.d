/**
 * site_container.d — Geometric site/chain storage for the tetrahedral lattice.
 *
 * Stores sites (R³ positions) and chains (ordered sequences of sites along
 * BC helices). Each site sits at the intersection of one R-chain and one
 * L-chain. Chains are parameterized by a ChainOrigin that gives the
 * analytic helix formula for computing positions and directions.
 *
 * This module knows nothing about spinors, τ operators, or walk physics.
 * It is purely geometric.
 */
module site_container;

import geometry : Vec3, Mat3, dot, norm,
    ChainOrigin, computeChainOrigin, chainCentroid, chainExitDir, chainVertexDirs;

/// Chirality flags for R/L helix chains.
enum bool IS_R = true;
enum bool IS_L = false;

/// The vertex permutation that generates the perpendicular cross-chain.
/// L-chain vertex sequence = R-chain vertices permuted by [1,3,0,2].
private immutable int[4] CROSS_PERM = [1, 3, 0, 2];
private immutable int[4] CROSS_PERM_INV = [2, 0, 3, 1];

/// Per-site geometric data.
struct Site {
    Vec3 pos;
    int rChain = -1, rIdx = -1;
    int lChain = -1, lIdx = -1;
}

/// A single entry in a chain: just the site ID.
/// (Operator matrices are stored separately by the walk code.)
struct ChainEntry {
    int siteId;
}

/// Deque backed by a flat array. Active elements are buf[lo .. hi].
struct Deque(T) {
    T[] buf;
    int lo, hi;

    @property int length() const { return hi - lo; }
    ref inout(T) opIndex(int i) inout { return buf[lo + i]; }

    void clear() { lo = cast(int)(buf.length / 2); hi = lo; }

    void pushBack(T val) {
        if (hi >= buf.length) expand();
        buf[hi++] = val;
    }
    void pushFront(T val) {
        if (lo <= 0) expand();
        buf[--lo] = val;
    }

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

    private enum DEQUE_INIT_CAP = 32;

    private void expand() {
        int n = length;
        int newCap = (buf.length < DEQUE_INIT_CAP / 2) ? DEQUE_INIT_CAP
                     : cast(int)(buf.length * 2);
        auto newBuf = new T[newCap];
        int newLo = (newCap - n) / 2;
        if (n > 0)
            newBuf[newLo .. newLo + n] = buf[lo .. hi];
        buf = newBuf;
        lo = newLo;
        hi = newLo + n;
    }
}

/// A geometric chain: ordered sequence of sites along a BC helix.
struct GeoChain {
    bool isR;
    bool isClosed;
    int rootSite;
    int rootIdx;
    Deque!ChainEntry entries;
    ChainOrigin origin;

    @property int length() const { return entries.length; }
    int siteAt(int i) const { return entries[i].siteId; }
}

/// The site container: sites + chains, purely geometric.
struct SiteContainer {
    Site[] sites;
    GeoChain[] chains;
    int nsites;
    int capacity;

    static SiteContainer create(int initCap) {
        SiteContainer sc;
        sc.capacity = initCap;
        sc.sites = new Site[initCap];
        sc.nsites = 0;
        return sc;
    }

    /// Allocate a new site at position pos. Returns site ID.
    int allocSite(Vec3 pos) {
        if (nsites >= capacity) {
            capacity *= 2;
            sites.length = capacity;
        }
        int id = nsites++;
        sites[id] = Site(pos);
        return id;
    }

    // ---- Chain access ----

    int chainNext(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx < 0) return -1;
        int nextIdx = idx + 1;
        if (nextIdx >= chains[chainId].length) return -1;
        return chains[chainId].siteAt(nextIdx);
    }

    int chainPrev(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx <= 0) return -1;
        return chains[chainId].siteAt(idx - 1);
    }

    bool hasChain(int siteId, bool isR) const {
        return (isR ? sites[siteId].rChain : sites[siteId].lChain) >= 0;
    }

    Vec3 exitDirForSite(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        if (chainId < 0) return Vec3(0, 0, 0);
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        return chainExitDir(&chains[chainId].origin, chains[chainId].rootIdx + idx);
    }

    // ---- Chain building ----

    /// Create a new chain from 4 initial vertices.
    /// The root site must already exist.
    int makeChain(int rootSite, bool isR, Vec3[4] initialVerts) {
        Vec3 c0 = (initialVerts[0] + initialVerts[1] + initialVerts[2] + initialVerts[3]) * 0.25;
        Vec3[4] dirs;
        foreach (k; 0 .. 4) {
            Vec3 d = initialVerts[k] - c0;
            double n = norm(d);
            if (n > 1e-15) dirs[k] = d * (1.0 / n);
        }

        int chainId = cast(int) chains.length;
        GeoChain newChain;
        newChain.isR = isR;
        newChain.rootSite = rootSite;
        newChain.rootIdx = 0;
        newChain.origin = computeChainOrigin(sites[rootSite].pos, dirs, 0, isR);

        ChainEntry rootEntry;
        rootEntry.siteId = rootSite;
        newChain.entries.pushBack(rootEntry);
        chains ~= newChain;

        if (isR) { sites[rootSite].rChain = chainId; sites[rootSite].rIdx = 0; }
        else     { sites[rootSite].lChain = chainId; sites[rootSite].lIdx = 0; }

        return chainId;
    }

    /// Extend a chain forward by one new site. Returns new site ID.
    int chainAppend(int chainId) {
        auto ch = &chains[chainId];
        int newChainIdx = ch.rootIdx + ch.length;
        Vec3 p = chainCentroid(&ch.origin, newChainIdx);
        int newSite = allocSite(p);

        int idx = ch.length;
        if (ch.isR) { sites[newSite].rChain = chainId; sites[newSite].rIdx = idx; }
        else        { sites[newSite].lChain = chainId; sites[newSite].lIdx = idx; }

        ChainEntry entry;
        entry.siteId = newSite;
        ch.entries.pushBack(entry);
        return newSite;
    }

    /// Extend a chain backward by one new site. Returns new site ID.
    int chainPrepend(int chainId) {
        auto ch = &chains[chainId];
        ch.rootIdx--;
        Vec3 p = chainCentroid(&ch.origin, ch.rootIdx);
        int newSite = allocSite(p);

        if (ch.isR) { sites[newSite].rChain = chainId; sites[newSite].rIdx = 0; }
        else        { sites[newSite].lChain = chainId; sites[newSite].lIdx = 0; }

        // Shift existing site indices
        for (int i = 0; i < ch.entries.length; i++) {
            int existingId = ch.entries[i].siteId;
            if (ch.isR) sites[existingId].rIdx++;
            else        sites[existingId].lIdx++;
        }

        ChainEntry entry;
        entry.siteId = newSite;
        ch.entries.pushFront(entry);
        return newSite;
    }

    /// Create the perpendicular cross-chain at a site.
    int makeCrossChain(int siteId, bool newIsR) {
        bool existingIsR = !newIsR;
        int chainId = existingIsR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = existingIsR ? sites[siteId].rIdx : sites[siteId].lIdx;
        int chainIdx = chains[chainId].rootIdx + idx;

        Vec3 c = chainCentroid(&chains[chainId].origin, chainIdx);
        Vec3[4] dirs = chainVertexDirs(&chains[chainId].origin, chainIdx);
        Vec3[4] existingVerts;
        foreach (k; 0 .. 4)
            existingVerts[k] = Vec3(c.x + dirs[k].x, c.y + dirs[k].y, c.z + dirs[k].z);

        auto perm = existingIsR ? CROSS_PERM : CROSS_PERM_INV;
        Vec3[4] crossVerts;
        foreach (k; 0 .. 4)
            crossVerts[k] = existingVerts[perm[k]];

        return makeChain(siteId, newIsR, crossVerts);
    }
}
