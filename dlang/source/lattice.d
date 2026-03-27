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
import geometry : Vec3, Mat3, dot, norm, helixStep, reorth, initTet, REORTH_INTERVAL, buildAllA4Rotations;
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

/// BC helix face patterns.
immutable int[4] PAT_R = [1, 3, 0, 2];
immutable int[4] PAT_L = [0, 2, 1, 3];

/// Perpendicularity mapping: given an R-chain face, return the L-chain face
/// that ensures disjoint face pairs (each walker step on exactly one chain).
/// R pair {rFace, prevFace(PAT_R, rFace)} and L pair {lFace, prevFace(PAT_L, lFace)}
/// must partition all 4 faces.
int perpFace(bool fromR, int face) {
    // Mapping: 0↔1, 2↔3
    immutable int[4] map = [1, 0, 3, 2];
    return map[face];
}

/// Tolerance for detecting degenerate cross products (near-parallel vectors).
enum double DEGEN_TOL = 1e-10;

int nextFace(const int[4] pat, int curFace) {
    foreach (i; 0 .. 4)
        if (pat[i] == curFace) return pat[(i + 1) % 4];
    return pat[0];
}

int prevFace(const int[4] pat, int curFace) {
    foreach (i; 0 .. 4)
        if (pat[i] == curFace) return pat[(i + 3) % 4];
    return pat[0];
}

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
    int rootSite;
    int rootIdx;
    Deque!(SiteOps!hasCoin) ops;
}

/// Per-site data.
struct Site {
    Vec3 pos;
    Vec3[4] dirs;
    int rChain = -1, rIdx = -1;
    int lChain = -1, lIdx = -1;
    int rFace = -1, lFace = -1;
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

    enum int INIT_CAPACITY = 1024;

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
        // Per chain: isR, rootSite, rootIdx, siteId list
        bool[] chainIsR;
        int[] chainRootSite;
        int[] chainRootIdx;
        int[][] chainSiteIds;
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
        foreach (i, ref ch; chains) {
            snap.chainIsR[i] = ch.isR;
            snap.chainRootSite[i] = ch.rootSite;
            snap.chainRootIdx[i] = ch.rootIdx;
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

    int allocSite(Vec3 pos, Vec3[4] dirs) {
        int id;
        if (freeCount > 0) id = freeList[--freeCount];
        else if (nsites < capacity) id = nsites++;
        else if (grow()) id = nsites++;
        else return -1;  // out of RAM — caller must handle gracefully
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
        return chains[chainId].ops[idx - 1].siteId;
    }

    int chainFace(int siteId, bool isR) const {
        return isR ? sites[siteId].rFace : sites[siteId].lFace;
    }

    void setChainFace(int siteId, bool isR, int face) {
        if (isR) sites[siteId].rFace = face; else sites[siteId].lFace = face;
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

        int face = isR ? sites[siteId].rFace : sites[siteId].lFace;
        Mat4 tau = makeTau(sites[siteId].dirs[face]);

        if (nextSite >= 0) {
            int nf = isR ? sites[nextSite].rFace : sites[nextSite].lFace;
            Mat4 tauNext = makeTau(sites[nextSite].dirs[nf]);
            op.fwdBlock = mul(frameTransport(tau, tauNext), projPlus(tau));
        }
        if (prevSite >= 0) {
            int pf = isR ? sites[prevSite].rFace : sites[prevSite].lFace;
            Mat4 tauPrev = makeTau(sites[prevSite].dirs[pf]);
            op.bwdBlock = mul(frameTransport(tau, tauPrev), projMinus(tau));
        }

        static if (hasCoin) {
            Vec3 e = sites[siteId].dirs[face];
            auto coins = buildDualParityCoins(e, coinCt, coinSt);
            op.coin1 = coins[0];
            op.coin2 = coins[1];
        }

        return op;
    }

    void chainAppend(int chainId, int siteId) {
        auto ch = &chains[chainId];
        bool isR = ch.isR;

        int prevSite = (ch.ops.length > 0) ? ch.ops[ch.ops.length - 1].siteId : -1;
        auto op = buildOps(siteId, prevSite, -1, isR);
        ch.ops.pushBack(op);

        // Update previous site's fwdBlock (it now has a successor)
        if (ch.ops.length >= 2) {
            int prevIdx = ch.ops.length - 2;
            int ps = ch.ops[prevIdx].siteId;
            int pf = isR ? sites[ps].rFace : sites[ps].lFace;
            Mat4 tauPrev = makeTau(sites[ps].dirs[pf]);
            int sf = isR ? sites[siteId].rFace : sites[siteId].lFace;
            Mat4 tauNew = makeTau(sites[siteId].dirs[sf]);
            ch.ops[prevIdx].fwdBlock = mul(frameTransport(tauPrev, tauNew), projPlus(tauPrev));
        }

        int idx = ch.ops.length - 1;
        if (isR) { sites[siteId].rChain = chainId; sites[siteId].rIdx = idx; }
        else     { sites[siteId].lChain = chainId; sites[siteId].lIdx = idx; }
    }

    void chainPrepend(int chainId, int siteId) {
        auto ch = &chains[chainId];
        bool isR = ch.isR;

        int nextSite = (ch.ops.length > 0) ? ch.ops[0].siteId : -1;
        auto op = buildOps(siteId, -1, nextSite, isR);
        ch.ops.pushFront(op);

        // Update next site's bwdBlock (it now has a predecessor)
        if (ch.ops.length >= 2) {
            int ns = ch.ops[1].siteId;
            int nf = isR ? sites[ns].rFace : sites[ns].lFace;
            Mat4 tauNext = makeTau(sites[ns].dirs[nf]);
            int sf = isR ? sites[siteId].rFace : sites[siteId].lFace;
            Mat4 tauNew = makeTau(sites[siteId].dirs[sf]);
            ch.ops[1].bwdBlock = mul(frameTransport(tauNext, tauNew), projMinus(tauNext));
        }

        // Shift existing indices
        for (int i = 1; i < ch.ops.length; i++) {
            int existingId = ch.ops[i].siteId;
            if (isR) sites[existingId].rIdx++;
            else     sites[existingId].lIdx++;
        }
        ch.rootIdx++;

        if (isR) { sites[siteId].rChain = chainId; sites[siteId].rIdx = 0; }
        else     { sites[siteId].lChain = chainId; sites[siteId].lIdx = 0; }
    }
}

// ---- Site generation ----

private struct ChainSeed { int siteId; bool isR; }

/// Create a new chain with the given root site.  Returns the chain ID.
/// If face >= 0, use that face; otherwise default to pat[0].
private int makeChain(bool hasCoin)(ref Lattice!hasCoin lat, int rootSite, bool isR, int face = -1) {
    const int[4] pat = isR ? PAT_R : PAT_L;
    if (face < 0) {
        // If the site already has the other chirality's chain, use the
        // perpendicularity mapping to determine this chain's starting face.
        int otherFace = lat.chainFace(rootSite, !isR);
        if (otherFace >= 0)
            face = perpFace(!isR, otherFace);
        else
            face = pat[0];
    }
    lat.setChainFace(rootSite, isR, face);

    int chainId = cast(int) lat.chains.length;
    Chain!hasCoin newChain;
    newChain.isR = isR;
    newChain.rootSite = rootSite;
    newChain.rootIdx = 0;
    SiteOps!hasCoin rootOp;
    rootOp.siteId = rootSite;
    static if (hasCoin) {
        Vec3 e = lat.sites[rootSite].dirs[lat.chainFace(rootSite, isR)];
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

/// Per-site orbit membership.
struct SiteOrbitInfo { int orbitIdx = -1; int memberIdx = -1; }
/// One member of an A4 orbit.
struct OrbitMember { int siteId; int rotIdx; }
/// An A4 orbit of sites.
struct OrbitData { OrbitMember[12] members; int size; }

int generateSites(bool hasCoin)(ref Lattice!hasCoin lat, double sigma, double seedThresh,
                  ref ProximityGrid grid) {
    import std.math : sqrt;
    grid.sitesPtr = &lat.sites;
    auto a4rots = buildAllA4Rotations();
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

    // Helper: extend a chain by n steps, return array of new site IDs.
    int[] extendN(int chainId, bool forward, int n) {
        int[] result;
        auto ch = &lat.chains[chainId];
        bool isR = ch.isR;
        const int[4] pat = isR ? PAT_R : PAT_L;
        int endId = forward ? ch.ops[ch.ops.length - 1].siteId : ch.ops[0].siteId;
        int curFace = lat.chainFace(endId, isR);
        Vec3 p = lat.sites[endId].pos;
        Vec3[4] d = lat.sites[endId].dirs;

        foreach (_; 0 .. n) {
            int stepFace = forward ? curFace : prevFace(pat, curFace);
            helixStep(p, d, stepFace);
            int nbFace = forward ? nextFace(pat, curFace) : stepFace;
            Vec3[4] dd = d; reorth(dd);

            // Check if a site already exists here (shared walker edge)
            int nb = grid.findSiteNear(p, mateTol);
            if (nb < 0) {
                nb = lat.allocSite(p, dd);
                grid.add(p, nb);
            }
            // Only add to this chain if the site isn't already on a chain
            // for this chirality (shared walker edges are on one chain only).
            if (lat.chainFace(nb, isR) < 0) {
                lat.setChainFace(nb, isR, nbFace);
                if (forward) lat.chainAppend(chainId, nb);
                else         lat.chainPrepend(chainId, nb);
            }
            curFace = nbFace;
            result ~= nb;
            ch = &lat.chains[chainId];  // refresh after potential realloc
        }
        return result;
    }

    int nChains = 0;

    // ==== Bootstrap: depth 0, 1, 2 (17 sites) ====

    // Depth 0: origin
    auto d0 = initTet();
    int origin = lat.allocSite(Vec3(0, 0, 0), d0);
    grid.add(Vec3(0, 0, 0), origin);

    // Origin's R and L chains, extended 2 steps each direction
    int rCh = makeChain!hasCoin(lat, origin, IS_R); nChains++;
    int lCh = makeChain!hasCoin(lat, origin, IS_L); nChains++;
    auto rFwd = extendN(rCh, true, 2);   // [depth1_A, depth2_B]
    auto rBwd = extendN(rCh, false, 2);  // [depth1_C, depth2_D]
    auto lFwd = extendN(lCh, true, 2);   // [depth1_E, depth2_F]
    auto lBwd = extendN(lCh, false, 2);  // [depth1_G, depth2_H]

    // Cross-chirality chains from depth-1 sites, extended 1 step each direction
    int[4] d1sites = [rFwd[0], rBwd[0], lFwd[0], lBwd[0]];
    int[] depth2cross;
    foreach (s; d1sites) {
        bool parentIsR = lat.sites[s].rChain >= 0 &&
                         lat.chains[lat.sites[s].rChain].rootSite != s;
        // If s is on origin's R-chain, cross-chirality is L; if on L-chain, cross is R.
        bool crossIsR = (lat.sites[s].lChain < 0);  // needs L? then crossIsR=false
        if (lat.chainFace(s, !crossIsR) >= 0)
            crossIsR = !crossIsR;  // pick whichever chirality s doesn't have
        // Actually simpler: s has one chirality from origin's chain; create the other
        bool needR = lat.chainFace(s, IS_R) < 0;
        int cc = makeChain!hasCoin(lat, s, needR); nChains++;
        auto fwd = extendN(cc, true, 1);
        auto bwd = extendN(cc, false, 1);
        depth2cross ~= fwd[0];
        depth2cross ~= bwd[0];
    }

    // Collect depth-2 sites
    int[] depth2 = [rFwd[1], rBwd[1], lFwd[1], lBwd[1]];
    depth2 ~= depth2cross;  // 4 + 8 = 12 sites

    // Create cross-chirality chains for all depth-2 sites (just roots, no extension)
    foreach (s; depth2) {
        if (lat.chainFace(s, IS_R) < 0) { makeChain!hasCoin(lat, s, IS_R); nChains++; }
        if (lat.chainFace(s, IS_L) < 0) { makeChain!hasCoin(lat, s, IS_L); nChains++; }
    }
    // Also ensure depth-1 sites have both chiralities
    foreach (s; d1sites) {
        if (lat.chainFace(s, IS_R) < 0) { makeChain!hasCoin(lat, s, IS_R); nChains++; }
        if (lat.chainFace(s, IS_L) < 0) { makeChain!hasCoin(lat, s, IS_L); nChains++; }
    }

    // Debug: print chain info for all bootstrap sites
    {
        import std.stdio : stderr;
        stderr.writefln("\n--- Bootstrap chain info ---");
        foreach (s; 0 .. lat.nsites) {
            int rCh2 = lat.sites[s].rChain;
            int lCh2 = lat.sites[s].lChain;
            int rLen = rCh2 >= 0 ? lat.chains[rCh2].ops.length : 0;
            int lLen = lCh2 >= 0 ? lat.chains[lCh2].ops.length : 0;
            stderr.writefln("  site %2d: rChain=%d rLen=%d rFace=%d  lChain=%d lLen=%d lFace=%d",
                s, rCh2, rLen, lat.chainFace(s, IS_R),
                lCh2, lLen, lat.chainFace(s, IS_L));
        }
    }

    // ==== Register orbits ====

    // Orbit 0: origin (size 1)
    addOrbit([origin], [0], 1);

    // Orbit 1: depth-1 sites (size 4) — find A4 rotations by position matching
    {
        int[4] ids = d1sites;
        int[4] rots;
        Vec3 canP = lat.sites[ids[0]].pos;  // canonical = first depth-1 site
        foreach (i; 0 .. 4) {
            Vec3 sp = lat.sites[ids[i]].pos;
            int bestRi = 0; double bestD = double.max;
            foreach (ri; 0 .. 12) {
                Vec3 cand = a4rots[ri].apply(canP);
                Vec3 dv = cand - sp; double d2 = dot(dv, dv);
                if (d2 < bestD) { bestD = d2; bestRi = ri; }
            }
            rots[i] = bestRi;
        }
        addOrbit(ids[], rots[], 4);
    }

    // Orbit 2: depth-2 sites (size 12)
    {
        int[12] ids;
        int[12] rots;
        foreach (i; 0 .. 12) ids[i] = depth2[i];
        Vec3 canP = lat.sites[ids[0]].pos;
        foreach (i; 0 .. 12) {
            Vec3 sp = lat.sites[ids[i]].pos;
            int bestRi = 0; double bestD = double.max;
            foreach (ri; 0 .. 12) {
                Vec3 cand = a4rots[ri].apply(canP);
                Vec3 dv = cand - sp; double d2 = dot(dv, dv);
                if (d2 < bestD) { bestD = d2; bestRi = ri; }
            }
            rots[i] = bestRi;
        }
        addOrbit(ids[], rots[], 12);
    }

    // ==== Orbit-driven frontier ====
    // Each entry is an orbit index.  Processing creates all 12 child sites
    // (from up to 4 primary directions), registers them in the grid, then
    // finds each child's predecessor and connects it to the right chain.
    int[] frontier;
    frontier ~= 2;  // depth-2 orbit

    // ---- Orbit-driven growth ----
    int fIdx = 0;
    while (fIdx < frontier.length) {
        int orbIdx = frontier[fIdx++];
        auto orb = &orbits[orbIdx];
        if (orb.size == 0) continue;

        int x0 = orb.members[0].siteId;
        int rj = orb.members[0].rotIdx;

        // Ensure primary has both chains
        if (lat.chainFace(x0, IS_R) < 0) { makeChain!hasCoin(lat, x0, IS_R); nChains++; }
        if (lat.chainFace(x0, IS_L) < 0) { makeChain!hasCoin(lat, x0, IS_L); nChains++; }

        bool anyExtended = false;

        // Try all 4 directions from the primary to find viable child positions
        foreach (tryR; [true, false]) {
            foreach (tryFwd; [true, false]) {
                const int[4] pat = tryR ? PAT_R : PAT_L;
                int x0Ch = tryR ? lat.sites[x0].rChain : lat.sites[x0].lChain;
                auto ch0 = &lat.chains[x0Ch];
                int x0End = tryFwd ? ch0.ops[ch0.ops.length - 1].siteId
                                   : ch0.ops[0].siteId;
                if (x0End != x0) continue;

                int curFace = lat.chainFace(x0, tryR);
                Vec3 p = lat.sites[x0].pos;
                Vec3[4] d = lat.sites[x0].dirs;
                int stepFace = tryFwd ? curFace : prevFace(pat, curFace);
                helixStep(p, d, stepFace);

                if (exp(-dot(p, p) / (2 * sigma * sigma)) < seedThresh) continue;
                if (grid.isTooClose(p)) continue;
                if (isRamLow()) continue;

                Vec3[4] dd = d;
                reorth(dd);

                // Canonical child position
                Vec3 canP = a4rots[rj].applyTranspose(p);
                Vec3[4] canD;
                foreach (fi; 0 .. 4) canD[fi] = a4rots[rj].applyTranspose(dd[fi]);

                // Phase 1: Create all 12 child sites and register in grid
                int[12] newIds;  int[12] newRots;  int newN = 0;
                Vec3[12] newPos;
                bool oom = false;

                foreach (ri; 0 .. 12) {
                    Vec3 yiP = a4rots[ri].apply(canP);
                    if (grid.isTooClose(yiP)) continue;

                    Vec3[4] yiD;
                    foreach (fi; 0 .. 4) yiD[fi] = a4rots[ri].apply(canD[fi]);

                    int yi = lat.allocSite(yiP, yiD);
                    if (yi < 0) { oom = true; break; }
                    grid.add(yiP, yi);

                    newIds[newN] = yi;
                    newRots[newN] = ri;
                    newPos[newN] = yiP;
                    newN++;
                }
                if (oom) break;

                // Phase 2: Connect each child to its predecessor's chain.
                // For each Y_i, compute 4 reverse helix steps to find the
                // predecessor site, then append/prepend Y_i to that chain.
                foreach (ni; 0 .. newN) {
                    int yi = newIds[ni];
                    Vec3 yiP = newPos[ni];
                    Vec3[4] yiD = lat.sites[yi].dirs;

                    // Try all 4 reverse steps: predecessor = Y_i.pos + (2/3)*Y_i.dirs[face]
                    // (helix step subtracts, so reverse adds)
                    bool connected = false;
                    foreach (eR; [true, false]) {
                        if (connected) break;
                        const int[4] ePat = eR ? PAT_R : PAT_L;
                        foreach (eFwd; [true, false]) {
                            if (connected) break;
                            // If Y_i is at face f on this chain, predecessor
                            // stepped through some face to reach Y_i.
                            // Forward: pred stepped through pred's curFace, giving
                            //   Y_i with nbFace = nextFace(pat, pred's curFace).
                            //   So pred's curFace = prevFace(pat, Y_i's face).
                            //   Pred is at Y_i.pos + (2/3)*Y_i.dirs[prevFace(pat, Y_i's face)]
                            //   ... but Y_i doesn't have a face for this chirality yet.
                            //
                            // Simpler: just compute where each face step leads FROM Y_i
                            // and check if an existing site is there. If the existing site
                            // is at a chain end, Y_i is its successor.

                            // Compute the helix step from Y_i through a face
                            // and check if a predecessor exists at the step position.
                            // Actually, the predecessor is at the REVERSE position.
                            // helixStep: pos' = pos - (2/3)*dirs[face]
                            // So predecessor pos = Y_i.pos + (2/3)*yiD[face] ... but
                            // yiD is the REFLECTED dirs (after the step). The predecessor's
                            // dirs before the step are reflect(yiD, yiD[face]).
                            //
                            // Let's just try matching: compute a step from Y_i in each
                            // direction and see if it lands on an existing site at a chain end.
                            // If site Z is one step from Y_i AND Z is at a chain end, then
                            // Z is Y_i's predecessor and Y_i should be appended/prepended.

                            // For the 4-way search we need Y_i's face for each chirality.
                            // We don't know it yet. But we can try each face.
                            foreach (face; 0 .. 4) {
                                Vec3 predP = yiP;
                                Vec3[4] predD = yiD;
                                helixStep(predP, predD, face);
                                // predP is one step FROM Y_i — it's the position of
                                // a potential neighbor of Y_i (not the predecessor).
                                // The PREDECESSOR is the site that stepped TO Y_i.
                                // predecessor.pos + step = Y_i.pos
                                // step = -(2/3)*predecessor.dirs[stepFace]
                                // predecessor.pos = Y_i.pos + (2/3)*predecessor.dirs[stepFace]
                                //
                                // After the step, dirs get reflected:
                                // Y_i.dirs = reflect(predecessor.dirs, predecessor.dirs[stepFace])
                                // So predecessor.dirs[stepFace] = -Y_i.dirs[stepFace] (reflection of itself)
                                // Wait: reflect(v, v) = v - 2(v·v)v = v - 2v = -v. Yes!
                                // predecessor.dirs[stepFace] = -yiD[face] (if face == stepFace)
                                //
                                // So predecessor.pos = Y_i.pos + (2/3)*(-yiD[face])
                                //                    = Y_i.pos - (2/3)*yiD[face]
                                // But that's the same as helixStep from Y_i through face!
                                // (helixStep: pos = pos - (2/3)*dirs[face])
                                // So predP = Y_i.pos - (2/3)*yiD[face] = the step FROM Y_i.
                                //
                                // This means: stepping FROM Y_i through face gives the
                                // predecessor position. The predecessor stepped through
                                // the same face index to reach Y_i (because reflection
                                // maps dirs[face] → -dirs[face], reversing the step).

                                int pred = grid.findSiteNear(predP, mateTol);
                                if (pred < 0) continue;
                                if (pred == yi) continue;

                                // Check pred is at a chain end and the chain edge
                                // through this face is available
                                foreach (pR; [true, false]) {
                                    if (connected) break;
                                    int pFace = lat.chainFace(pred, pR);
                                    if (pFace < 0) continue;
                                    const int[4] pPat = pR ? PAT_R : PAT_L;

                                    // Check forward: pred's curFace step leads to Y_i?
                                    if (pFace == face) {
                                        // Forward step from pred through pFace
                                        int pCh = pR ? lat.sites[pred].rChain : lat.sites[pred].lChain;
                                        auto pch = &lat.chains[pCh];
                                        if (pch.ops[pch.ops.length - 1].siteId == pred) {
                                            int yiFace = nextFace(pPat, pFace);
                                            lat.setChainFace(yi, pR, yiFace);
                                            lat.chainAppend(pCh, yi);
                                            if (lat.chainFace(yi, !pR) < 0) {
                                                makeChain!hasCoin(lat, yi, !pR);
                                                nChains++;
                                            }
                                            connected = true;
                                        }
                                    }

                                    // Check backward: prevFace(pPat, pFace) == face?
                                    if (!connected && prevFace(pPat, pFace) == face) {
                                        int pCh = pR ? lat.sites[pred].rChain : lat.sites[pred].lChain;
                                        auto pch = &lat.chains[pCh];
                                        if (pch.ops[0].siteId == pred) {
                                            int yiFace = face;  // backward: nbFace = stepFace
                                            lat.setChainFace(yi, pR, yiFace);
                                            lat.chainPrepend(pCh, yi);
                                            if (lat.chainFace(yi, !pR) < 0) {
                                                makeChain!hasCoin(lat, yi, !pR);
                                                nChains++;
                                            }
                                            connected = true;
                                        }
                                    }
                                }
                                if (connected) break;
                            }
                        }
                    }
                    // If not connected, the site exists but isn't linked yet.
                    // It will be connected when its predecessor's chain catches up.
                }

                if (newN > 0) {
                    int newOrb = addOrbit(newIds[0 .. newN], newRots[0 .. newN], newN);
                    frontier ~= newOrb;
                    anyExtended = true;
                }
            }
        }

        if (anyExtended)
            frontier ~= orbIdx;
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
    foreach (f; 0 .. 4) {
        assert(prevFace(PAT_R, nextFace(PAT_R, f)) == f);
        assert(nextFace(PAT_R, prevFace(PAT_R, f)) == f);
    }
}

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
    assert(lat.chainFace(0, true) >= 0);
    assert(lat.chainFace(0, false) >= 0);

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
