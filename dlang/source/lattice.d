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
import geometry : Vec3, dot, norm, helixStep, reorth, initTet, REORTH_INTERVAL;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, alpha;

/// Chirality flags for R/L helix chains.
enum bool IS_R = true;
enum bool IS_L = false;

/// BC helix face patterns.
immutable int[4] PAT_R = [1, 3, 0, 2];
immutable int[4] PAT_L = [0, 1, 2, 3];

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

/// Density grid for limiting site count per unit volume.
struct DensityGrid {
    /// Side length of each cubic grid cell.
    enum double GRID_CELL_SIZE = 1.0;
    /// Maximum grid dimension along each axis.
    enum int MAX_GRID_DIM = 500;

    int[] counts;
    double gridHalf;
    int gridN;
    int maxPerCell;

    static DensityGrid create(double halfExtent, int maxPer) {
        DensityGrid g;
        g.gridHalf = halfExtent;
        g.gridN = cast(int)(2.0 * halfExtent / GRID_CELL_SIZE) + 1;
        if (g.gridN > MAX_GRID_DIM) g.gridN = MAX_GRID_DIM;
        g.counts = new int[g.gridN * g.gridN * g.gridN];
        g.counts[] = 0;
        g.maxPerCell = maxPer;
        return g;
    }

    int idx(Vec3 pos) const {
        int gx = cast(int)((pos.x + gridHalf) / GRID_CELL_SIZE);
        int gy = cast(int)((pos.y + gridHalf) / GRID_CELL_SIZE);
        int gz = cast(int)((pos.z + gridHalf) / GRID_CELL_SIZE);
        if (gx < 0) gx = 0; if (gx >= gridN) gx = gridN - 1;
        if (gy < 0) gy = 0; if (gy >= gridN) gy = gridN - 1;
        if (gz < 0) gz = 0; if (gz >= gridN) gz = gridN - 1;
        return gx * gridN * gridN + gy * gridN + gz;
    }

    bool isFull(Vec3 pos) const { return counts[idx(pos)] >= maxPerCell; }
    void increment(Vec3 pos) { counts[idx(pos)]++; }
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
    int maxSites;
    int[] freeList;
    int freeCount;

    // Coin parameters (only meaningful when hasCoin)
    double coinCt = 1, coinSt = 0;

    static Lattice create(int maxSites) {
        Lattice lat;
        lat.maxSites = maxSites;
        lat.sites = new Site[maxSites];
        lat.psiRe = new double[4 * maxSites]; lat.psiRe[] = 0;
        lat.psiIm = new double[4 * maxSites]; lat.psiIm[] = 0;
        lat.tmpRe = new double[4 * maxSites]; lat.tmpRe[] = 0;
        lat.tmpIm = new double[4 * maxSites]; lat.tmpIm[] = 0;
        lat.nsites = 0;
        lat.freeList = new int[maxSites];
        lat.freeCount = 0;
        return lat;
    }

    int allocSite(Vec3 pos, Vec3[4] dirs) {
        int id;
        if (freeCount > 0) id = freeList[--freeCount];
        else if (nsites < maxSites) id = nsites++;
        else return -1;  // capacity full — caller must handle gracefully
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

int generateSites(bool hasCoin)(ref Lattice!hasCoin lat, double sigma, double seedThresh,
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

        int chainId = cast(int) lat.chains.length;
        Chain!hasCoin newChain;
        newChain.isR = isR;
        newChain.rootSite = rootSite;
        newChain.rootIdx = 0;
        // Push root with no neighbors yet
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

        int fwd = extendDir!hasCoin(lat, chainId, true, sigma, seedThresh, grid, queue, qTail);
        int bwd = extendDir!hasCoin(lat, chainId, false, sigma, seedThresh, grid, queue, qTail);

        if (fwd + bwd > 0) nChains++;
    }

    // Reserve extra capacity for runtime chain extension
    enum CHAIN_RESERVE_PAD = 7;
    foreach (ref ch; lat.chains) {
        int extra = ch.ops.length + CHAIN_RESERVE_PAD;
        ch.ops.reserve(extra);
    }

    return nChains;
}

private int extendDir(bool hasCoin)(ref Lattice!hasCoin lat, int chainId, bool forward,
                      double sigma, double seedThresh,
                      ref DensityGrid grid,
                      ChainSeed[] queue, ref int qTail) {
    auto ch = &lat.chains[chainId];
    bool isR = ch.isR;
    const int[4] pat = isR ? PAT_R : PAT_L;

    int endSite = forward ? ch.ops[ch.ops.length - 1].siteId : ch.ops[0].siteId;
    int curFace = lat.chainFace(endSite, isR);
    Vec3 p = lat.sites[endSite].pos;
    Vec3[4] d = lat.sites[endSite].dirs;

    int created = 0;

    for (int step = 0; step < lat.maxSites; step++) {
        int stepFace = forward ? curFace : prevFace(pat, curFace);
        helixStep(p, d, stepFace);
        if ((step + 1) % REORTH_INTERVAL == 0) reorth(d);

        if (exp(-dot(p, p) / (2 * sigma * sigma)) < seedThresh) break;
        if (grid.isFull(p)) break;

        int nbFace = forward ? nextFace(pat, curFace) : stepFace;

        Vec3[4] dd = d;
        reorth(dd);
        int nb = lat.allocSite(p, dd);
        if (nb < 0) break;  // lattice full
        lat.setChainFace(nb, isR, nbFace);
        grid.increment(p);

        if (forward)
            lat.chainAppend(chainId, nb);
        else
            lat.chainPrepend(chainId, nb);

        queue[qTail++] = ChainSeed(nb, !isR);

        curFace = nbFace;
        created++;

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
    auto lat = Lattice!false.create(100000);
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
