/**
 * lattice.d — Tetrahedral lattice: chain-based site storage and generation.
 *
 * Sites live at the intersection of R and L helix chains. Each chain stores
 * a deque of SiteOps entries — one per site — containing the site ID and
 * precomputed operator matrices (shift blocks, and optionally coins).
 *
 * The hasCoin template parameter controls whether coin matrices are stored.
 * When false (e.g. theta=0, V-mixing only), no coin storage is allocated.
 */
module lattice;

import std.math : sqrt, exp, fabs, cos, sin;
import geometry : Vec3, Mat3, dot, norm, initTet, buildAllA4Rotations,
    ChainOrigin, computeChainOrigin, chainCentroid, chainExitDir, chainVertexDirs;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, alpha;
import core.sys.linux.sys.sysinfo : sysinfo, sysinfo_;

/// Minimum free RAM (bytes) below which the simulation aborts.  Default 4 GB.
enum ulong MIN_FREE_RAM = 4UL * 1024 * 1024 * 1024;

/// Returns available free RAM in bytes, or ulong.max on failure.
ulong freeRamBytes() {
    sysinfo_ info;
    if (sysinfo(&info) != 0) return ulong.max;
    return info.freeram * info.mem_unit;
}

/// Returns true if free RAM is critically low (below MIN_FREE_RAM).
/// Returns false if RAM is adequate or cannot be determined.
bool isRamLow() {
    ulong avail = freeRamBytes();
    return avail != ulong.max && avail < MIN_FREE_RAM;
}

/// Chirality flags for R/L helix chains.
enum bool IS_R = true;
enum bool IS_L = false;

/// Tolerance for detecting degenerate cross products (near-parallel vectors).
enum double DEGEN_TOL = 1e-10;

/// Deque backed by a flat array. Active elements are buf[lo .. hi].
struct Deque(T) {
    T[] buf;
    int lo, hi;

    @property int length() const { return hi - lo; }
    ref inout(T) opIndex(int i) inout { return buf[lo + i]; }

    /// Reset to empty, reusing existing buffer.
    void clear() { lo = cast(int)(buf.length / 2); hi = lo; }

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
    private enum DEQUE_GROW_FACTOR = 2;

    private void expand() {
        int n = length;
        int newCap = (buf.length < DEQUE_INIT_CAP / 2) ? DEQUE_INIT_CAP
                     : cast(int)(buf.length * DEQUE_GROW_FACTOR);
        auto newBuf = new T[newCap];
        int newLo = (newCap - n) / 2;
        if (n > 0)
            newBuf[newLo .. newLo + n] = buf[lo .. hi];
        buf = newBuf;
        lo = newLo;
        hi = newLo + n;
    }
}

/// Per-site operator data, stored contiguously in the chain deque.
struct SiteOps(bool hasCoin) {
    int siteId;
    Mat4 fwdBlock;   /// applied to ψ(this site), result to next site
    Mat4 bwdBlock;   /// applied to ψ(this site), result to prev site
    static if (hasCoin) {
        Mat4 coin1;  /// first coin matrix (f₁·α or e·α)
        Mat4 coin2;  /// second coin matrix (f₂·α), dual parity only
    }
}

/// Build coin matrices for a site given its chain direction.
Mat4[2] buildDualParityCoins(Vec3 e, double ct, double st) {
    double en = norm(e);
    Vec3 ehat = e * (1.0 / en);

    Vec3 f1 = Vec3(ehat.y, -ehat.x, 0);
    double fn = norm(f1);
    if (fn < DEGEN_TOL) { f1 = Vec3(-ehat.z, 0, ehat.x); fn = norm(f1); }
    f1 = f1 * (1.0 / fn);

    Vec3 f2 = Vec3(ehat.y*f1.z - ehat.z*f1.y,
                    ehat.z*f1.x - ehat.x*f1.z,
                    ehat.x*f1.y - ehat.y*f1.x);
    fn = norm(f2);
    f2 = f2 * (1.0 / fn);

    Mat4[2] coins;
    coins[0] = buildCoinMatrix([f1.x, f1.y, f1.z], ct, st);
    coins[1] = buildCoinMatrix([f2.x, f2.y, f2.z], ct, st);
    return coins;
}

/// Build coin matrix: exp(-iθ(d·α)) = cos(θ)I - i sin(θ)(d·α)
Mat4 buildCoinMatrix(double[3] d, double ct, double st) {
    Mat4 m;
    foreach (a; 0 .. 4)
        foreach (b; 0 .. 4) {
            double eaRe = 0, eaIm = 0;
            foreach (k; 0 .. 3) {
                auto al = alpha(k);
                eaRe += d[k] * al.re[4*a+b];
                eaIm += d[k] * al.im[4*a+b];
            }
            m.re[4*a+b] = (a == b ? ct : 0.0) + st * eaIm;
            m.im[4*a+b] = -st * eaRe;
        }
    return m;
}

/// A helix chain: a deque of SiteOps entries.
struct Chain(bool hasCoin) {
    bool isR;
    bool isClosed;        /// true for periodic chains (manifold walk)
    int rootSite;
    int rootIdx;
    Deque!(SiteOps!hasCoin) ops;
    ChainOrigin origin;  /// analytic helix parameters for this chain
}

/// Per-site data.
struct Site {
    Vec3 pos;
    int rChain = -1, rIdx = -1;
    int lChain = -1, lIdx = -1;
}

/// Spatial hash for minimum-distance site proximity checks.
/// Rejects new sites that are within dMin of any existing site.
/// Uses a hash map of cell keys → linked-list heads in a flat pool,
/// so memory is proportional to the number of sites, not grid volume.
struct ProximityGrid {
    /// Linked-list node stored in a flat pool.
    struct Node { int siteId; int next; }  // next = -1 for end of list

    int[long] heads;   // cell key → index into pool (head of list), or absent
    Node[] pool;       // flat pool of nodes
    int poolCount;

    double gridHalf;
    double cellSize;
    int gridN;
    double dMin2;      // squared minimum distance
    Site[]* sitesPtr;   // pointer to lattice's sites slice (stays valid after grow)

    enum int POOL_INIT_CAP = 1024;

    static ProximityGrid create(double halfExtent, double dMin) {
        ProximityGrid g;
        g.gridHalf = halfExtent;
        g.cellSize = dMin;  // cell size = dMin so we only check 27 neighbors
        g.gridN = cast(int)(2.0 * halfExtent / g.cellSize) + 1;
        g.dMin2 = dMin * dMin;
        g.pool = new Node[POOL_INIT_CAP];
        g.poolCount = 0;
        return g;
    }

    private void cellCoords(Vec3 pos, out int gx, out int gy, out int gz) const {
        gx = cast(int)((pos.x + gridHalf) / cellSize);
        gy = cast(int)((pos.y + gridHalf) / cellSize);
        gz = cast(int)((pos.z + gridHalf) / cellSize);
        if (gx < 0) gx = 0; if (gx >= gridN) gx = gridN - 1;
        if (gy < 0) gy = 0; if (gy >= gridN) gy = gridN - 1;
        if (gz < 0) gz = 0; if (gz >= gridN) gz = gridN - 1;
    }

    private long cellKey(int gx, int gy, int gz) const {
        return cast(long) gx * gridN * gridN + cast(long) gy * gridN + gz;
    }

    /// Returns true if pos is within dMin of any existing site.
    bool isTooClose(Vec3 pos) const {
        int cx, cy, cz;
        cellCoords(pos, cx, cy, cz);

        // Check 3x3x3 neighborhood
        foreach (dx; -1 .. 2)
            foreach (dy; -1 .. 2)
                foreach (dz; -1 .. 2) {
                    int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                    if (nx < 0 || nx >= gridN || ny < 0 || ny >= gridN ||
                        nz < 0 || nz >= gridN) continue;
                    long key = cellKey(nx, ny, nz);
                    auto p = key in heads;
                    if (p is null) continue;
                    int idx = *p;
                    while (idx >= 0) {
                        int id = pool[idx].siteId;
                        auto s = (*sitesPtr)[id];
                        Vec3 d = Vec3(pos.x - s.pos.x,
                                      pos.y - s.pos.y,
                                      pos.z - s.pos.z);
                        if (d.x*d.x + d.y*d.y + d.z*d.z < dMin2)
                            return true;
                        idx = pool[idx].next;
                    }
                }
        return false;
    }

    /// Find the nearest site within tol of pos. Returns site ID or -1.
    int findSiteNear(Vec3 pos, double tol) const {
        int cx, cy, cz;
        cellCoords(pos, cx, cy, cz);
        double tol2 = tol * tol;
        int bestId = -1;
        double bestDist2 = tol2;

        foreach (dx; -1 .. 2)
            foreach (dy; -1 .. 2)
                foreach (dz; -1 .. 2) {
                    int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                    if (nx < 0 || nx >= gridN || ny < 0 || ny >= gridN ||
                        nz < 0 || nz >= gridN) continue;
                    long key = cellKey(nx, ny, nz);
                    auto pp = key in heads;
                    if (pp is null) continue;
                    int idx = *pp;
                    while (idx >= 0) {
                        int id = pool[idx].siteId;
                        auto s = (*sitesPtr)[id];
                        Vec3 dd = Vec3(pos.x - s.pos.x,
                                       pos.y - s.pos.y,
                                       pos.z - s.pos.z);
                        double d2 = dd.x*dd.x + dd.y*dd.y + dd.z*dd.z;
                        if (d2 < bestDist2) {
                            bestDist2 = d2;
                            bestId = id;
                        }
                        idx = pool[idx].next;
                    }
                }
        return bestId;
    }

    /// Register a site in the grid.
    void add(Vec3 pos, int siteId) {
        if (poolCount >= pool.length)
            pool.length = pool.length * 2;
        int cx, cy, cz;
        cellCoords(pos, cx, cy, cz);
        long key = cellKey(cx, cy, cz);
        int nodeIdx = poolCount++;
        auto p = key in heads;
        pool[nodeIdx] = Node(siteId, p is null ? -1 : *p);
        heads[key] = nodeIdx;
    }
}


/// The lattice, templated on whether coins are precomputed.
struct Lattice(bool hasCoin) {
    alias Ops = SiteOps!hasCoin;
    alias ChainT = Chain!hasCoin;

    Site[] sites;
    double[] psiRe, psiIm;
    double[] tmpRe, tmpIm;
    ChainT[] chains;
    int nsites;
    int capacity;    // current allocation size (grows on demand)
    int[] freeList;
    int freeCount;

    // Coin parameters (only meaningful when hasCoin)
    double coinCt = 1, coinSt = 0;

    enum int INIT_CAPACITY = 2 * 1024 * 1024;  // 2M sites

    static Lattice create(int initCap = INIT_CAPACITY) {
        Lattice lat;
        lat.capacity = initCap;
        lat.sites = new Site[initCap];
        lat.psiRe = new double[4 * initCap]; lat.psiRe[] = 0;
        lat.psiIm = new double[4 * initCap]; lat.psiIm[] = 0;
        lat.tmpRe = new double[4 * initCap]; lat.tmpRe[] = 0;
        lat.tmpIm = new double[4 * initCap]; lat.tmpIm[] = 0;
        lat.nsites = 0;
        lat.freeList = new int[initCap];
        lat.freeCount = 0;
        return lat;
    }

    /// Double the lattice capacity if RAM allows. Returns false if out of memory.
    bool grow() {
        import std.stdio : stderr;
        int newCap = capacity * 2;
        ulong growBytes = (newCap - capacity) * (Site.sizeof + 4UL * 4 * double.sizeof + int.sizeof);
        ulong avail = freeRamBytes();
        if (avail != ulong.max && growBytes + MIN_FREE_RAM > avail) {
            stderr.writefln("  RAM limit: cannot grow lattice from %d to %d sites (%d MB needed, %d MB free)",
                            capacity, newCap, growBytes / (1024 * 1024), avail / (1024 * 1024));
            return false;
        }

        sites.length = newCap;
        psiRe.length = 4 * newCap; psiRe[4 * capacity .. $] = 0;
        psiIm.length = 4 * newCap; psiIm[4 * capacity .. $] = 0;
        tmpRe.length = 4 * newCap; tmpRe[4 * capacity .. $] = 0;
        tmpIm.length = 4 * newCap; tmpIm[4 * capacity .. $] = 0;
        freeList.length = newCap;
        capacity = newCap;
        return true;
    }

    /// Lightweight snapshot: sites + chain site-ID lists (no operator matrices).
    struct Snapshot {
        Site[] sites;
        int nsites;
        int[] freeList;
        int freeCount;
        // Per chain: isR, rootSite, rootIdx, siteId list, origin
        bool[] chainIsR;
        int[] chainRootSite;
        int[] chainRootIdx;
        int[][] chainSiteIds;
        ChainOrigin[] chainOrigins;
    }

    Snapshot takeSnapshot() const {
        Snapshot snap;
        snap.sites = sites[0 .. nsites].dup;
        snap.nsites = nsites;
        snap.freeList = freeList[0 .. freeCount].dup;
        snap.freeCount = freeCount;
        int nc = cast(int) chains.length;
        snap.chainIsR = new bool[nc];
        snap.chainRootSite = new int[nc];
        snap.chainRootIdx = new int[nc];
        snap.chainSiteIds = new int[][nc];
        snap.chainOrigins = new ChainOrigin[nc];
        foreach (i, ref ch; chains) {
            snap.chainIsR[i] = ch.isR;
            snap.chainRootSite[i] = ch.rootSite;
            snap.chainRootIdx[i] = ch.rootIdx;
            snap.chainOrigins[i] = ch.origin;
            auto ids = new int[ch.ops.length];
            foreach (j; 0 .. ch.ops.length)
                ids[j] = ch.ops[j].siteId;
            snap.chainSiteIds[i] = ids;
        }
        return snap;
    }

    /// Restore from snapshot: rebuild sites, chains, and recompute operator blocks.
    void restoreSnapshot(ref const Snapshot snap) {
        nsites = snap.nsites;
        sites[0 .. nsites] = snap.sites[];
        freeCount = snap.freeCount;
        if (freeCount > 0)
            freeList[0 .. freeCount] = snap.freeList[];

        int nc = cast(int) snap.chainSiteIds.length;
        chains.length = nc;
        foreach (ci; 0 .. nc) {
            auto ids = snap.chainSiteIds[ci];
            bool isR = snap.chainIsR[ci];
            chains[ci].isR = isR;
            chains[ci].rootSite = snap.chainRootSite[ci];
            chains[ci].rootIdx = snap.chainRootIdx[ci];
            chains[ci].origin = snap.chainOrigins[ci];

            // Clear and refill ops, reusing existing deque buffer
            int n = cast(int) ids.length;
            chains[ci].ops.clear();
            foreach (j; 0 .. n) {
                int prev = (j > 0) ? ids[j - 1] : -1;
                int next = (j < n - 1) ? ids[j + 1] : -1;
                auto op = buildOps(ids[j], prev, next, isR);
                chains[ci].ops.pushBack(op);
            }

            // Update site chain/index references
            foreach (j; 0 .. n) {
                if (isR) { sites[ids[j]].rChain = ci; sites[ids[j]].rIdx = j; }
                else     { sites[ids[j]].lChain = ci; sites[ids[j]].lIdx = j; }
            }
        }

        psiRe[0 .. 4 * nsites] = 0;
        psiIm[0 .. 4 * nsites] = 0;
        tmpRe[0 .. 4 * nsites] = 0;
        tmpIm[0 .. 4 * nsites] = 0;
    }

    int allocSite(Vec3 pos) {
        int id;
        if (freeCount > 0) id = freeList[--freeCount];
        else if (nsites < capacity) id = nsites++;
        else if (grow()) id = nsites++;
        else return -1;  // out of RAM — caller must handle gracefully
        sites[id] = Site(pos);
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
        auto sr = psiRe; psiRe = tmpRe; tmpRe = sr;
        auto si = psiIm; psiIm = tmpIm; tmpIm = si;
    }

    // ---- Chain access ----

    int chainNext(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx < 0) return -1;
        int nextIdx = idx + 1;
        if (nextIdx >= chains[chainId].ops.length) return -1;
        return chains[chainId].ops[nextIdx].siteId;
    }

    int chainPrev(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx <= 0) return -1;
        assert(idx < chains[chainId].ops.length,
               "chainPrev: site idx out of range");
        return chains[chainId].ops[idx - 1].siteId;
    }

    /// Get the exit direction for site siteId on the given chirality chain.
    Vec3 exitDirForSite(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        if (chainId < 0) return Vec3(0, 0, 0);
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        return chainExitDir(&chains[chainId].origin, chains[chainId].rootIdx + idx);
    }

    /// Check if site has a chain of given chirality.
    bool hasChain(int siteId, bool isR) const {
        return (isR ? sites[siteId].rChain : sites[siteId].lChain) >= 0;
    }

    /// Create the perpendicular cross-chain at a site that already has one chain.
    /// Uses the [1,3,0,2] vertex permutation to generate the perpendicular helix.
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

        return makeChainFromVertices!hasCoin(this, siteId, newIsR, crossVerts);
    }

    /// Get the SiteOps for a site on its chain.
    const(Ops)* siteOps(int siteId, bool isR) const {
        int chainId = isR ? sites[siteId].rChain : sites[siteId].lChain;
        int idx = isR ? sites[siteId].rIdx : sites[siteId].lIdx;
        if (chainId < 0 || idx < 0) return null;
        return &chains[chainId].ops[idx];
    }

    // ---- Chain modification ----

    /// Build a SiteOps for a site, computing shift blocks (and coins if hasCoin).
    /// prevSite is the neighbor toward which the backward block points (-1 if none).
    /// nextSite is the neighbor toward which the forward block points (-1 if none).
    private Ops buildOps(int siteId, int prevSite, int nextSite, bool isR) {
        Ops op;
        op.siteId = siteId;

        Mat4 tau = makeTau(exitDirForSite(siteId, isR));

        if (nextSite >= 0) {
            Mat4 tauNext = makeTau(exitDirForSite(nextSite, isR));
            op.fwdBlock = mul(frameTransport(tau, tauNext), projPlus(tau));
        }
        if (prevSite >= 0) {
            Mat4 tauPrev = makeTau(exitDirForSite(prevSite, isR));
            op.bwdBlock = mul(frameTransport(tau, tauPrev), projMinus(tau));
        }

        static if (hasCoin) {
            Vec3 e = exitDirForSite(siteId, isR);
            auto coins = buildDualParityCoins(e, coinCt, coinSt);
            op.coin1 = coins[0];
            op.coin2 = coins[1];
        }

        return op;
    }

    void chainAppend(int chainId, int siteId) {
        auto ch = &chains[chainId];
        bool isR = ch.isR;

        // Link site to chain BEFORE buildOps so exitDirForSite works
        int idx = ch.ops.length;
        if (isR) { sites[siteId].rChain = chainId; sites[siteId].rIdx = idx; }
        else     { sites[siteId].lChain = chainId; sites[siteId].lIdx = idx; }

        int prevSite = (ch.ops.length > 0) ? ch.ops[ch.ops.length - 1].siteId : -1;
        auto op = buildOps(siteId, prevSite, -1, isR);
        ch.ops.pushBack(op);

        // Update previous site's fwdBlock (it now has a successor)
        if (ch.ops.length >= 2) {
            int prevIdx = ch.ops.length - 2;
            int ps = ch.ops[prevIdx].siteId;
            Mat4 tauPrev = makeTau(exitDirForSite(ps, isR));
            Mat4 tauNew = makeTau(exitDirForSite(siteId, isR));
            ch.ops[prevIdx].fwdBlock = mul(frameTransport(tauPrev, tauNew), projPlus(tauPrev));
        }
    }

    void chainPrepend(int chainId, int siteId) {
        auto ch = &chains[chainId];
        bool isR = ch.isR;

        // Decrement rootIdx and link new site at idx=0.
        // Formula index for new site = (rootIdx-1) + 0 = rootIdx-1. ✓
        ch.rootIdx--;
        if (isR) { sites[siteId].rChain = chainId; sites[siteId].rIdx = 0; }
        else     { sites[siteId].lChain = chainId; sites[siteId].lIdx = 0; }

        // Shift existing site indices FIRST so that exitDirForSite returns
        // the correct formula index for all sites.  After rootIdx-- and
        // rIdx++, each existing site's formula index = (rootIdx-1) + (old_idx+1)
        // = rootIdx + old_idx, which is unchanged. ✓
        for (int i = 0; i < ch.ops.length; i++) {
            int existingId = ch.ops[i].siteId;
            if (isR) sites[existingId].rIdx++;
            else     sites[existingId].lIdx++;
        }

        // Now all exitDirForSite calls return correct values.
        int nextSite = (ch.ops.length > 0) ? ch.ops[0].siteId : -1;
        auto op = buildOps(siteId, -1, nextSite, isR);
        ch.ops.pushFront(op);

        // Update next site's bwdBlock (it now has a predecessor)
        if (ch.ops.length >= 2) {
            int ns = ch.ops[1].siteId;
            Mat4 tauNext = makeTau(exitDirForSite(ns, isR));
            Mat4 tauNew = makeTau(exitDirForSite(siteId, isR));
            ch.ops[1].bwdBlock = mul(frameTransport(tauNext, tauNew), projMinus(tauNext));
        }
    }

    /// Close a chain into a loop: connect the last site to the first.
    /// Sets isClosed=true and computes the wrap-around fwdBlock/bwdBlock.
    void closeChain(int chainId) {
        auto ch = &chains[chainId];
        int n = ch.ops.length;
        assert(n >= 2, "Cannot close a chain with fewer than 2 sites");

        ch.isClosed = true;

        // Last site's fwdBlock → wraps to first site
        int lastSite = ch.ops[n-1].siteId;
        int firstSite = ch.ops[0].siteId;
        bool isR = ch.isR;

        Mat4 tauLast = makeTau(exitDirForSite(lastSite, isR));
        Mat4 tauFirst = makeTau(exitDirForSite(firstSite, isR));

        ch.ops[n-1].fwdBlock = mul(frameTransport(tauLast, tauFirst), projPlus(tauLast));

        // First site's bwdBlock → wraps to last site
        ch.ops[0].bwdBlock = mul(frameTransport(tauFirst, tauLast), projMinus(tauFirst));
    }
}

// ---- Site generation ----

private struct ChainSeed { int siteId; bool isR; }

/// Compute the 3x3 rotation matrix for a vertex permutation.
/// R = (3/4) Σ_k tetDirs[perm[k]] ⊗ tetDirs[k] (same formula as A4 rotations).
Mat3 computeCrossRotation(const int[4] perm) {
    import geometry : tetDirs;
    Mat3 R;
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 3) {
            double s = 0;
            foreach (k; 0 .. 4) {
                double pi = void, qj = void;
                final switch (i) { case 0: pi=tetDirs[perm[k]].x; break; case 1: pi=tetDirs[perm[k]].y; break; case 2: pi=tetDirs[perm[k]].z; break; }
                final switch (j) { case 0: qj=tetDirs[k].x; break; case 1: qj=tetDirs[k].y; break; case 2: qj=tetDirs[k].z; break; }
                s += pi * qj;
            }
            R.m[3*i+j] = 0.75 * s;
        }
    return R;
}

/// The vertex permutation that generates the perpendicular cross-chain.
/// L-chain vertex sequence = R-chain vertices permuted by [1,3,0,2].
/// Inverse permutation (R from L) = [2,0,3,1].
private immutable int[4] CROSS_PERM = [1, 3, 0, 2];
private immutable int[4] CROSS_PERM_INV = [2, 0, 3, 1];

/// Create a new chain from 4 initial vertices in sequence order.
/// The chain is a BC helix: tet 0 = {v[0],v[1],v[2],v[3]}, drops v[0] first.
/// Returns the chain ID.
package int makeChainFromVertices(bool hasCoin)(ref Lattice!hasCoin lat, int rootSite,
                                                bool isR, Vec3[4] initialVerts) {
    // Compute ChainOrigin from initial vertices
    Vec3 c0 = (initialVerts[0] + initialVerts[1] + initialVerts[2] + initialVerts[3]) * 0.25;
    Vec3 exitDir = initialVerts[0] - c0;
    double en = norm(exitDir);
    if (en > 1e-15) exitDir = exitDir * (1.0 / en);

    // Build dirs array (normalized vertex directions from centroid)
    Vec3[4] dirs;
    foreach (k; 0 .. 4) {
        Vec3 d = initialVerts[k] - c0;
        double n = norm(d);
        if (n > 1e-15) dirs[k] = d * (1.0 / n);
    }

    // face = 0 since the exit direction is always dirs[0] (first vertex = dropped vertex)
    int face = 0;

    int chainId = cast(int) lat.chains.length;
    Chain!hasCoin newChain;
    newChain.isR = isR;
    newChain.rootSite = rootSite;
    newChain.rootIdx = 0;
    newChain.origin = computeChainOrigin(lat.sites[rootSite].pos, dirs, face, isR);
    SiteOps!hasCoin rootOp;
    rootOp.siteId = rootSite;
    static if (hasCoin) {
        Vec3 e = chainExitDir(&newChain.origin, 0);
        auto coins = buildDualParityCoins(e, lat.coinCt, lat.coinSt);
        rootOp.coin1 = coins[0];
        rootOp.coin2 = coins[1];
    }
    newChain.ops.pushBack(rootOp);
    lat.chains ~= newChain;

    if (isR) { lat.sites[rootSite].rChain = chainId; lat.sites[rootSite].rIdx = 0; }
    else     { lat.sites[rootSite].lChain = chainId; lat.sites[rootSite].lIdx = 0; }

    return chainId;
}

// makeCrossChain is defined as a Lattice method below (in the struct body).

/// Per-site orbit membership.
struct SiteOrbitInfo { int orbitIdx = -1; int memberIdx = -1; }
/// One member of an A4 orbit.
struct OrbitMember { int siteId; int rotIdx; }
/// An A4 orbit of sites.
struct OrbitData { OrbitMember[12] members; int size; }

/// Simple xorshift64 RNG for deterministic seed lattice generation.
struct Xorshift64 {
    ulong state;

    this(ulong seed) { state = seed == 0 ? 0x12345678_9ABCDEF0 : seed; }

    ulong next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    /// Returns a uniform double in [0, 1).
    double uniform01() {
        return (next() >> 11) * (1.0 / (1UL << 53));
    }
}

int generateSites(bool hasCoin)(ref Lattice!hasCoin lat, double sigma, double seedThresh,
                  ref ProximityGrid grid, ulong rngSeed = 0) {
    import std.math : sqrt;
    grid.sitesPtr = &lat.sites;
    auto a4rots = buildAllA4Rotations();

    // RNG for jittering the priority queue. seed=0 means no jitter (deterministic).
    bool useJitter = rngSeed != 0;
    auto rng = Xorshift64(rngSeed);
    // Jitter scale: fraction of step_length² — small enough to preserve radial order
    // at large scale, but shuffle entries at similar distances.
    import geometry : STEP_LEN;
    enum double JITTER_SCALE = 0.01 * STEP_LEN * STEP_LEN;  // ~0.0044
    double mateTol = sqrt(grid.dMin2) * 0.5;

    // ---- Orbit tracking ----
    SiteOrbitInfo[] siteInfo;
    OrbitData[] orbits;

    void ensureInfo(int id) {
        if (id >= siteInfo.length) siteInfo.length = (id + 1) * 2;
    }
    int addOrbit(int[] ids, int[] rots, int n) {
        OrbitData od; od.size = n;
        int oi = cast(int) orbits.length;
        foreach (i; 0 .. n) {
            od.members[i] = OrbitMember(ids[i], rots[i]);
            ensureInfo(ids[i]);
            siteInfo[ids[i]] = SiteOrbitInfo(oi, i);
        }
        orbits ~= od;
        return oi;
    }

    // Helper: extend a chain by n steps using analytic centroid formula.
    int[] extendN(int chainId, bool forward, int n) {
        int[] result;
        auto ch = &lat.chains[chainId];
        bool isR = ch.isR;

        foreach (_; 0 .. n) {
            int newChainIdx = forward
                ? ch.rootIdx + cast(int) ch.ops.length
                : ch.rootIdx - 1;
            Vec3 p = chainCentroid(&ch.origin, newChainIdx);

            int nb = grid.findSiteNear(p, mateTol);
            if (nb < 0) {
                nb = lat.allocSite(p);
                grid.add(p, nb);
            }
            if (!lat.hasChain(nb, isR)) {
                if (forward) lat.chainAppend(chainId, nb);
                else         lat.chainPrepend(chainId, nb);
            }
            result ~= nb;
        }
        return result;
    }

    int nChains = 0;

    // ==== Symmetric lattice growth ====
    //
    // Every site is on exactly one R-chain and one L-chain.
    // Growth is orbit-driven: extending one canonical chain by one step
    // creates 12 A4-symmetric copies, each with its cross-chain.
    //
    // Algorithm:
    // 1. Create seed: origin site with R-chain and L-chain
    // 2. Frontier = set of chain ends to extend
    // 3. Pick a canonical chain end, compute the next site position p
    // 4. If p is within the Gaussian envelope:
    //    a. Compute canonical position canP = A4_inv(p)
    //    b. Create 12 A4 copies of the new site
    //    c. For each copy: append to parent chain, create cross-chain
    //    d. Add new chain ends to frontier
    // 5. Repeat until frontier is empty

    import geometry : helixVertex, Mat3;

    // ---- Helper: create a site with both chains ----

    /// Apply A4 rotation to a ChainOrigin: rotates pos and frame.
    ChainOrigin rotateChainOrigin(const ChainOrigin* orig, const Mat3* rot) {
        ChainOrigin result;
        // Rotate the frame: new_rot = A4_rot * orig_rot
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 3) {
                double s = 0;
                foreach (k; 0 .. 3)
                    s += rot.m[3*i+k] * orig.rot.m[3*k+j];
                result.rot.m[3*i+j] = s;
            }
        result.pos0 = rot.apply(orig.pos0);
        result.tSign = orig.tSign;
        result.slotPerm = orig.slotPerm;
        result.facePat = orig.facePat;
        result.faceOffset = orig.faceOffset;
        return result;
    }

    /// Create a chain from a rotated version of a template chain's origin.
    int makeRotatedChain(bool hasCoin)(ref Lattice!hasCoin lat, int rootSite,
                                       const ChainOrigin* templateOrigin,
                                       const Mat3* rot, bool isR) {
        int chainId = cast(int) lat.chains.length;
        Chain!hasCoin newChain;
        newChain.isR = isR;
        newChain.rootSite = rootSite;
        newChain.rootIdx = 0;
        newChain.origin = rotateChainOrigin(templateOrigin, rot);
        // Fix pos0 to match actual site position (avoid drift)
        newChain.origin.pos0 = lat.sites[rootSite].pos;

        SiteOps!hasCoin rootOp;
        rootOp.siteId = rootSite;
        static if (hasCoin) {
            Vec3 e = chainExitDir(&newChain.origin, 0);
            auto coins = buildDualParityCoins(e, lat.coinCt, lat.coinSt);
            rootOp.coin1 = coins[0];
            rootOp.coin2 = coins[1];
        }
        newChain.ops.pushBack(rootOp);
        lat.chains ~= newChain;

        if (isR) { lat.sites[rootSite].rChain = chainId; lat.sites[rootSite].rIdx = 0; }
        else     { lat.sites[rootSite].lChain = chainId; lat.sites[rootSite].lIdx = 0; }

        return chainId;
    }

    // Pre-allocate chains array to avoid reallocation during growth.
    lat.chains.reserve(lat.capacity * 3);

    // ---- Seed: origin with one R-chain and one L-chain ----
    int origin = lat.allocSite(Vec3(0, 0, 0));
    grid.add(Vec3(0, 0, 0), origin);
    addOrbit([origin], [0], 1);

    Vec3[4] seedVerts;
    foreach (k; 0 .. 4) seedVerts[k] = helixVertex(k);
    int seedR = makeChainFromVertices!hasCoin(lat, origin, IS_R, seedVerts); nChains++;
    int seedL = lat.makeCrossChain(origin, IS_L); nChains++;

    // ---- Frontier: priority queue of site orbits, ordered by distance ----
    // Each entry is a site orbit. We pick a representative, find a chain end
    // to extend, and create the 12 A4-symmetric copies.
    struct FrontierEntry {
        int orbitIdx;
        int chainId;   // which chain at the representative site to extend
        bool forward;
        double dist2;
    }

    FrontierEntry[] heap;

    void heapPush(FrontierEntry e) {
        heap ~= e;
        int i = cast(int) heap.length - 1;
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (heap[parent].dist2 <= heap[i].dist2) break;
            auto tmp = heap[parent]; heap[parent] = heap[i]; heap[i] = tmp;
            i = parent;
        }
    }

    FrontierEntry heapPop() {
        auto result = heap[0];
        heap[0] = heap[heap.length - 1];
        heap.length--;
        int i = 0;
        while (true) {
            int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
            if (left < heap.length && heap[left].dist2 < heap[smallest].dist2)
                smallest = left;
            if (right < heap.length && heap[right].dist2 < heap[smallest].dist2)
                smallest = right;
            if (smallest == i) break;
            auto tmp = heap[smallest]; heap[smallest] = heap[i]; heap[i] = tmp;
            i = smallest;
        }
        return result;
    }

    /// Find a chain end at site s that can be extended. Returns the chain ID
    /// and direction (forward/backward), or chainId=-1 if no extension possible.
    /// Also returns the position of the proposed extension.
    struct ExtensionCandidate {
        int chainId;
        bool forward;
        Vec3 pos;
        double dist2;
    }

    ExtensionCandidate findExtension(int siteId) {
        ExtensionCandidate best;
        best.chainId = -1;
        best.dist2 = double.max;

        foreach (isR; [true, false]) {
            int chainId = isR ? lat.sites[siteId].rChain : lat.sites[siteId].lChain;
            if (chainId < 0) continue;
            int idx = isR ? lat.sites[siteId].rIdx : lat.sites[siteId].lIdx;
            int n = lat.chains[chainId].ops.length;

            // Check forward end
            if (idx == n - 1) {
                int nextIdx = lat.chains[chainId].rootIdx + n;
                Vec3 p = chainCentroid(&lat.chains[chainId].origin, nextIdx);
                double d2 = dot(p, p);
                if (d2 < best.dist2) {
                    best = ExtensionCandidate(chainId, true, p, d2);
                }
            }

            // Check backward end
            if (idx == 0) {
                int prevIdx = lat.chains[chainId].rootIdx - 1;
                Vec3 p = chainCentroid(&lat.chains[chainId].origin, prevIdx);
                double d2 = dot(p, p);
                if (d2 < best.dist2) {
                    best = ExtensionCandidate(chainId, false, p, d2);
                }
            }
        }
        return best;
    }

    /// At partner site X_i, find the chain + direction whose extension matches
    /// the target position (= A4 rotation of the canonical extension position).
    ExtensionCandidate findMatchingExtension(int siteId, Vec3 targetPos) {
        ExtensionCandidate best;
        best.chainId = -1;
        best.dist2 = double.max;

        foreach (isR; [true, false]) {
            int chainId = isR ? lat.sites[siteId].rChain : lat.sites[siteId].lChain;
            if (chainId < 0) continue;
            int idx = isR ? lat.sites[siteId].rIdx : lat.sites[siteId].lIdx;
            int n = lat.chains[chainId].ops.length;

            // Check forward end
            if (idx == n - 1) {
                int nextIdx = lat.chains[chainId].rootIdx + n;
                Vec3 p = chainCentroid(&lat.chains[chainId].origin, nextIdx);
                Vec3 dv = Vec3(p.x - targetPos.x, p.y - targetPos.y, p.z - targetPos.z);
                double d2 = dv.x*dv.x + dv.y*dv.y + dv.z*dv.z;
                if (d2 < best.dist2) {
                    best = ExtensionCandidate(chainId, true, p, d2);
                }
            }

            // Check backward end
            if (idx == 0) {
                int prevIdx = lat.chains[chainId].rootIdx - 1;
                Vec3 p = chainCentroid(&lat.chains[chainId].origin, prevIdx);
                Vec3 dv = Vec3(p.x - targetPos.x, p.y - targetPos.y, p.z - targetPos.z);
                double d2 = dv.x*dv.x + dv.y*dv.y + dv.z*dv.z;
                if (d2 < best.dist2) {
                    best = ExtensionCandidate(chainId, false, p, d2);
                }
            }
        }
        return best;
    }

    /// Add ALL extendable chain ends from a site orbit to the frontier.
    /// Applies random jitter to the priority key for diversity (if rngSeed != 0).
    void addOrbitToFrontier(int orbitIdx) {
        auto orb = &orbits[orbitIdx];
        if (orb.size == 0) return;
        int x0 = orb.members[0].siteId;
        foreach (isR; [true, false]) {
            int chainId = isR ? lat.sites[x0].rChain : lat.sites[x0].lChain;
            if (chainId < 0) continue;
            int idx = isR ? lat.sites[x0].rIdx : lat.sites[x0].lIdx;
            int n = lat.chains[chainId].ops.length;
            if (idx == n - 1) {
                int nextIdx = lat.chains[chainId].rootIdx + n;
                Vec3 p = chainCentroid(&lat.chains[chainId].origin, nextIdx);
                double d2 = dot(p, p);
                if (useJitter) d2 += JITTER_SCALE * rng.uniform01();
                heapPush(FrontierEntry(orbitIdx, chainId, true, d2));
            }
            if (idx == 0) {
                int prevIdx = lat.chains[chainId].rootIdx - 1;
                Vec3 p = chainCentroid(&lat.chains[chainId].origin, prevIdx);
                double d2 = dot(p, p);
                if (useJitter) d2 += JITTER_SCALE * rng.uniform01();
                heapPush(FrontierEntry(orbitIdx, chainId, false, d2));
            }
        }
    }

    addOrbitToFrontier(0);  // seed: origin orbit

    // ---- Main growth loop ----
    while (heap.length > 0) {
        auto entry = heapPop();
        auto orb = &orbits[entry.orbitIdx];
        if (orb.size == 0) continue;

        // Step 1: Use the specific chain/direction from the frontier entry
        int x0 = orb.members[0].siteId;
        int ext0chain = entry.chainId;
        bool ext0fwd = entry.forward;

        // Verify this chain end is still extendable (may have been extended by another entry)
        {
            int idx = lat.sites[x0].rChain == ext0chain ? lat.sites[x0].rIdx : lat.sites[x0].lIdx;
            int n = lat.chains[ext0chain].ops.length;
            if (ext0fwd && idx != n - 1) continue;
            if (!ext0fwd && idx != 0) continue;
        }

        int nextIdx = ext0fwd
            ? lat.chains[ext0chain].rootIdx + lat.chains[ext0chain].ops.length
            : lat.chains[ext0chain].rootIdx - 1;
        Vec3 canP = chainCentroid(&lat.chains[ext0chain].origin, nextIdx);

        // Gaussian envelope check
        if (exp(-dot(canP, canP) / (2 * sigma * sigma)) < seedThresh) continue;
        if (grid.isTooClose(canP)) continue;
        if (isRamLow()) continue;

        // Step 2: Create the canonical new site Y_0
        int y0 = lat.allocSite(canP);
        if (y0 < 0) continue;
        grid.add(canP, y0);

        if (ext0fwd) lat.chainAppend(ext0chain, y0);
        else         lat.chainPrepend(ext0chain, y0);
        lat.makeCrossChain(y0, !lat.chains[ext0chain].isR);
        nChains++;

        // Step 3: Create A4 partners Y_1..Y_11
        int[12] newSites;
        int[12] newRots;
        int newCount = 0;
        newSites[newCount] = y0;
        newRots[newCount] = 0;
        newCount++;

        int r0 = orb.members[0].rotIdx;  // A4 rotation of X_0

        foreach (mi; 1 .. orb.size) {
            int xi = orb.members[mi].siteId;
            int ri = orb.members[mi].rotIdx;

            // The target position for Y_i = A4_rot(ri) @ A4_rot(r0)^{-1} @ canP
            // Since A4 rotations are orthogonal, inverse = transpose
            Vec3 relP = a4rots[r0].applyTranspose(canP);
            Vec3 targetP = a4rots[ri].apply(relP);

            // Find which chain at X_i extends to targetP
            auto extI = findMatchingExtension(xi, targetP);
            if (extI.chainId < 0) continue;  // shouldn't happen
            assert(extI.dist2 < mateTol * mateTol,
                   "Partner extension position doesn't match target");

            int yi = lat.allocSite(targetP);
            if (yi < 0) continue;
            grid.add(targetP, yi);

            if (extI.forward) lat.chainAppend(extI.chainId, yi);
            else              lat.chainPrepend(extI.chainId, yi);
            lat.makeCrossChain(yi, !lat.chains[extI.chainId].isR);
            nChains++;

            newSites[newCount] = yi;
            newRots[newCount] = ri;
            newCount++;
        }

        // Register site orbit and add to frontier
        if (newCount > 0) {
            int newOrb = addOrbit(newSites[0 .. newCount], newRots[0 .. newCount], newCount);

            // Add both old and new orbits back to the frontier
            addOrbitToFrontier(entry.orbitIdx);
            addOrbitToFrontier(newOrb);
        }
    }

    // Reserve extra capacity for runtime chain extension
    enum CHAIN_RESERVE_PAD = 7;
    foreach (ref ch; lat.chains) {
        int extra = ch.ops.length + CHAIN_RESERVE_PAD;
        ch.ops.reserve(extra);
    }

    return nChains;
}

// ---- D unit tests ----

unittest {
    auto lat = Lattice!false.create(100000);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = ProximityGrid.create(maxChainLen * stepLen + 5.0, 0.35);
    int nChains = generateSites(lat, sigma, 1e-4, grid);

    assert(lat.nsites > 1);
    assert(nChains > 0);
    assert(lat.chains.length > 0);
    assert(lat.hasChain(0, true));
    assert(lat.hasChain(0, false));

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
    auto lat = Lattice!false.create(100);
    int id0 = lat.allocSite(Vec3(0, 0, 0));
    int id1 = lat.allocSite(Vec3(1, 0, 0));
    assert(lat.nsites == 2);
    lat.removeSite(id0);
    assert(lat.freeCount == 1);
    int id2 = lat.allocSite(Vec3(2, 0, 0));
    assert(id2 == id0);
    assert(lat.freeCount == 0);
}
