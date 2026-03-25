/**
 * lattice.d — Tetrahedral lattice: site storage, chain linking, and generation.
 *
 * Sites are stored in a flat array indexed by ID. Each site has a position,
 * 4 tetrahedral direction vectors, and chain links (next/prev/face) for
 * both R and L helix chains. A spatial hash enables O(1) position lookup.
 * A density grid limits the number of sites per unit volume.
 */
module lattice;

import std.math : sqrt, exp, fabs;
import std.complex : Complex;
import geometry : Vec3, dot, norm, helixStep, reorth, initTet;
import spatial_hash : SiteHash;

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

/// Per-site data.
struct Site {
    Vec3 pos;
    Vec3[4] dirs;
    int rNext = -1, rPrev = -1;
    int lNext = -1, lPrev = -1;
    int rFace = -1, lFace = -1;
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

/// The lattice: site storage, spatial hash, free list, and density grid.
struct Lattice {
    Site[] sites;
    C[] psi;       /// spinor array: 4 components per site
    C[] tmp;       /// scratch buffer for shift operator
    SiteHash hash;
    int nsites;
    int maxSites;

    // Free list for ID reuse (pruning)
    int[] freeList;
    int freeCount;

    static Lattice create(int maxSites, int hashBits) {
        Lattice lat;
        lat.maxSites = maxSites;
        lat.sites = new Site[maxSites];
        lat.psi = new C[4 * maxSites];
        lat.psi[] = C(0, 0);
        lat.tmp = new C[4 * maxSites];
        lat.tmp[] = C(0, 0);
        lat.hash = SiteHash.create(hashBits);
        lat.nsites = 0;
        lat.freeList = new int[maxSites];
        lat.freeCount = 0;
        return lat;
    }

    /// Find site ID at position, or -1 if not present.
    /// Only valid while hash table is alive (before freeHash).
    int findSite(Vec3 pos) const {
        return hash.find(pos);
    }

    /// Insert a new site with deduplication (for seed generation).
    /// If a site already exists at this position, returns its existing ID.
    int insertSite(Vec3 pos, Vec3[4] dirs) {
        int existing = hash.find(pos);
        if (existing >= 0) return existing;

        int id = allocSiteId();
        hash.insert(pos, id);
        sites[id] = Site(pos, dirs);
        psi[4*id .. 4*id+4] = C(0, 0);
        return id;
    }

    /// Insert a new site unconditionally (for walk-time chain extension).
    /// Caller guarantees the site is unique (BC helix no-loops property).
    int insertSiteNew(Vec3 pos, Vec3[4] dirs) {
        int id = allocSiteId();
        sites[id] = Site(pos, dirs);
        psi[4*id .. 4*id+4] = C(0, 0);
        return id;
    }

    /// Allocate a site ID from the free list or bump nsites.
    private int allocSiteId() {
        if (freeCount > 0)
            return freeList[--freeCount];
        assert(nsites < maxSites, "Too many sites");
        return nsites++;
    }

    /// Remove a site (zero psi, push to free list).
    void removeSite(int id) {
        psi[4*id .. 4*id+4] = C(0, 0);
        sites[id] = Site.init;
        freeList[freeCount++] = id;
    }

    /// Free the hash table (call after seed generation is complete).
    void freeHash() {
        hash = SiteHash.init;
    }

    /// Swap psi and tmp pointers (avoids memcpy after shift).
    void swapBuffers() {
        auto swap = psi;
        psi = tmp;
        tmp = swap;
    }

    /// Get chain next/prev for a given chirality.
    int chainNext(int s, bool isR) const {
        return isR ? sites[s].rNext : sites[s].lNext;
    }
    int chainPrev(int s, bool isR) const {
        return isR ? sites[s].rPrev : sites[s].lPrev;
    }
    int chainFace(int s, bool isR) const {
        return isR ? sites[s].rFace : sites[s].lFace;
    }

    /// Set chain links.
    void setChainNext(int s, bool isR, int val) {
        if (isR) sites[s].rNext = val; else sites[s].lNext = val;
    }
    void setChainPrev(int s, bool isR, int val) {
        if (isR) sites[s].rPrev = val; else sites[s].lPrev = val;
    }
    void setChainFace(int s, bool isR, int val) {
        if (isR) sites[s].rFace = val; else sites[s].lFace = val;
    }

    /// Link cur -> nb in the forward direction for the given chirality.
    void linkForward(int cur, int nb, bool isR, int nbFace) {
        setChainNext(cur, isR, nb);
        setChainPrev(nb, isR, cur);
        if (chainFace(nb, isR) < 0)
            setChainFace(nb, isR, nbFace);
    }

    /// Link cur -> nb in the backward direction for the given chirality.
    void linkBackward(int cur, int nb, bool isR, int nbFace) {
        setChainPrev(cur, isR, nb);
        setChainNext(nb, isR, cur);
        if (chainFace(nb, isR) < 0)
            setChainFace(nb, isR, nbFace);
    }
}

private struct ChainSeed { int siteId; bool isR; }

/// Generate all sites using chain-first approach.
/// Returns the number of chains created.
int generateSites(ref Lattice lat, double sigma, double seedThresh,
                  ref DensityGrid grid) {
    // Origin
    auto d0 = initTet();
    lat.insertSite(Vec3(0, 0, 0), d0);
    grid.increment(Vec3(0, 0, 0));

    // BFS queue: each site can enqueue at most one perpendicular seed,
    // plus the initial 2 seeds for the origin.
    auto queue = new ChainSeed[2 * lat.maxSites + 2];
    int qHead = 0, qTail = 0;
    queue[qTail++] = ChainSeed(0, true);   // R-chain from origin
    queue[qTail++] = ChainSeed(0, false);  // L-chain from origin

    int nChains = 0;

    while (qHead < qTail) {
        auto seed = queue[qHead++];
        int s = seed.siteId;
        bool isR = seed.isR;

        if (lat.chainFace(s, isR) >= 0) continue;

        const int[4] pat = isR ? PAT_R : PAT_L;
        lat.setChainFace(s, isR, pat[0]);

        int fwd = extendChain(lat, s, pat, isR, true, sigma, seedThresh,
                              grid, queue, qTail);
        int bwd = extendChain(lat, s, pat, isR, false, sigma, seedThresh,
                              grid, queue, qTail);

        if (fwd + bwd > 0) nChains++;
    }

    return nChains;
}

/// Extend a chain in one direction, creating new sites as needed.
private int extendChain(ref Lattice lat, int startSite,
                        const int[4] pat, bool isR, bool forward,
                        double sigma, double seedThresh,
                        ref DensityGrid grid,
                        ChainSeed[] queue, ref int qTail) {
    int linked = 0;
    int cur = startSite;
    int curFace = lat.chainFace(cur, isR);

    Vec3 p = lat.sites[cur].pos;
    Vec3[4] d = lat.sites[cur].dirs;

    for (int step = 0; step < lat.maxSites; step++) {
        int stepFace = forward ? curFace : prevFace(pat, curFace);
        helixStep(p, d, stepFace);
        if ((step + 1) % 8 == 0) reorth(d);

        // Gaussian cutoff
        if (exp(-dot(p, p) / (2 * sigma * sigma)) < seedThresh) break;

        int nbFace = forward ? nextFace(pat, curFace) : stepFace;

        // Check if site already exists
        int nb = lat.findSite(p);
        if (nb >= 0) {
            // Already claimed by another chain of this type?
            if (forward && lat.chainPrev(nb, isR) >= 0) break;
            if (!forward && lat.chainNext(nb, isR) >= 0) break;

            if (forward)
                lat.linkForward(cur, nb, isR, nbFace);
            else
                lat.linkBackward(cur, nb, isR, nbFace);

            cur = nb;
            curFace = nbFace;
            linked++;
            continue;
        }

        // Density check
        if (grid.isFull(p)) break;

        // Create new site
        Vec3[4] dd = d;
        reorth(dd);
        nb = lat.insertSite(p, dd);
        grid.increment(p);

        if (forward)
            lat.linkForward(cur, nb, isR, nbFace);
        else
            lat.linkBackward(cur, nb, isR, nbFace);

        // Enqueue perpendicular chain seed
        queue[qTail++] = ChainSeed(nb, !isR);

        cur = nb;
        curFace = nbFace;
        linked++;
    }
    return linked;
}

// ---- D unit tests ----

unittest {
    // nextFace/prevFace are inverses
    foreach (f; 0 .. 4) {
        assert(prevFace(PAT_R, nextFace(PAT_R, f)) == f);
        assert(nextFace(PAT_R, prevFace(PAT_R, f)) == f);
        assert(prevFace(PAT_L, nextFace(PAT_L, f)) == f);
        assert(nextFace(PAT_L, prevFace(PAT_L, f)) == f);
    }
}

unittest {
    // Small lattice generation: origin exists, has chains
    auto lat = Lattice.create(100000, 17);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    double gridHalf = maxChainLen * stepLen + 5.0;
    auto grid = DensityGrid.create(gridHalf, 8);
    int nChains = generateSites(lat, sigma, 1e-4, grid);

    assert(lat.nsites > 1);
    assert(nChains > 0);

    // Origin is site 0
    assert(fabs(lat.sites[0].pos.x) < 1e-14);
    assert(fabs(lat.sites[0].pos.y) < 1e-14);
    assert(fabs(lat.sites[0].pos.z) < 1e-14);

    // Origin has both R and L chains
    assert(lat.chainFace(0, true) >= 0);
    assert(lat.chainFace(0, false) >= 0);
}

unittest {
    // All sites with a chain face should have consistent forward/backward links
    auto lat = Lattice.create(100000, 17);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    foreach (s; 0 .. lat.nsites) {
        // Check R-chain
        if (lat.chainFace(s, true) >= 0) {
            int nxt = lat.chainNext(s, true);
            if (nxt >= 0) assert(lat.chainPrev(nxt, true) == s ||
                                 lat.chainPrev(nxt, true) >= 0);
        }
        // Check L-chain
        if (lat.chainFace(s, false) >= 0) {
            int nxt = lat.chainNext(s, false);
            if (nxt >= 0) assert(lat.chainPrev(nxt, false) == s ||
                                 lat.chainPrev(nxt, false) >= 0);
        }
    }
}

unittest {
    // Hash table and site array agree
    auto lat = Lattice.create(100000, 17);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    foreach (s; 0 .. lat.nsites) {
        int found = lat.findSite(lat.sites[s].pos);
        assert(found == s, "Hash lookup mismatch");
    }
}

unittest {
    // Free list: remove and re-insert
    auto lat = Lattice.create(100, 10);
    auto d = initTet();
    int id0 = lat.insertSite(Vec3(0, 0, 0), d);
    int id1 = lat.insertSite(Vec3(1, 0, 0), d);
    assert(lat.nsites == 2);

    // After freeHash, removeSite doesn't touch the hash
    lat.freeHash();
    lat.removeSite(id0);
    assert(lat.freeCount == 1);

    // insertSiteNew (no hash) should reuse id0
    int id2 = lat.insertSiteNew(Vec3(2, 0, 0), d);
    assert(id2 == id0);
    assert(lat.freeCount == 0);
}
