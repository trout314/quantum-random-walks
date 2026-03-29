/**
 * manifold_interop.d — C interface for building and running a manifold walk.
 *
 * Python creates the lattice structure (sites, closed chains) via these
 * functions, then runs the walk loop in D for performance.
 */
module manifold_interop;

import geometry : Vec3;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, matVecSplit;
import lattice : Lattice;
import operators : pureShift, applyVmix;

/// Opaque handle for the lattice.  We store a heap-allocated pointer
/// so the GC keeps it alive across C calls.
private Lattice!false* gLat;

// ---- Lattice lifecycle ----

export extern(C)
void manifold_create(int capacity) {
    gLat = new Lattice!false;
    *gLat = Lattice!false.create(capacity);
}

export extern(C)
int manifold_nsites() {
    return gLat.nsites;
}

export extern(C)
int manifold_nchains() {
    return cast(int) gLat.chains.length;
}

// ---- Site creation ----

/// Allocate a site at (x, y, z). Returns site ID.
export extern(C)
int manifold_alloc_site(double x, double y, double z) {
    return gLat.allocSite(Vec3(x, y, z));
}

// ---- Chain creation ----

/// Create a closed chain from a list of site IDs and per-site exit directions.
///
/// siteIds:  array of nSites site IDs in chain order
/// exitDirs: array of 3*nSites doubles (x,y,z for each site's exit direction)
/// isR:      true for R-chain, false for L-chain
///
/// Builds all fwdBlock/bwdBlock ops and closes the chain (wrap-around).
/// Links each site to this chain.
/// Returns the chain ID.
export extern(C)
int manifold_add_closed_chain(const(int)* siteIds, const(double)* exitDirs,
                              int nSites, bool isR) {
    alias Ops = Lattice!false.Ops;

    // Create chain
    int chainId = cast(int) gLat.chains.length;
    Lattice!false.ChainT newChain;
    newChain.isR = isR;
    newChain.isClosed = true;
    newChain.rootSite = siteIds[0];
    newChain.rootIdx = 0;

    // Build ops for each site
    foreach (i; 0 .. nSites) {
        int sid = siteIds[i];
        Vec3 d = Vec3(exitDirs[3*i], exitDirs[3*i+1], exitDirs[3*i+2]);
        Mat4 tau = makeTau(d);

        int prevIdx = (i > 0) ? i - 1 : nSites - 1;
        int nextIdx = (i < nSites - 1) ? i + 1 : 0;
        Vec3 dPrev = Vec3(exitDirs[3*prevIdx], exitDirs[3*prevIdx+1], exitDirs[3*prevIdx+2]);
        Vec3 dNext = Vec3(exitDirs[3*nextIdx], exitDirs[3*nextIdx+1], exitDirs[3*nextIdx+2]);
        Mat4 tauPrev = makeTau(dPrev);
        Mat4 tauNext = makeTau(dNext);

        Ops op;
        op.siteId = sid;
        op.fwdBlock = mul(frameTransport(tau, tauNext), projPlus(tau));
        op.bwdBlock = mul(frameTransport(tau, tauPrev), projMinus(tau));
        newChain.ops.pushBack(op);

        // Link site to chain
        if (isR) { gLat.sites[sid].rChain = chainId; gLat.sites[sid].rIdx = i; }
        else     { gLat.sites[sid].lChain = chainId; gLat.sites[sid].lIdx = i; }
    }

    gLat.chains ~= newChain;
    return chainId;
}

// ---- Wavefunction access ----

/// Set spinor at site s: psi[s] = (re[0]+i*im[0], ..., re[3]+i*im[3])
export extern(C)
void manifold_set_psi(int siteId, const(double)* re, const(double)* im) {
    gLat.psiRe[4*siteId .. 4*siteId+4] = re[0..4];
    gLat.psiIm[4*siteId .. 4*siteId+4] = im[0..4];
}

/// Read all psi into caller-provided buffers (4*nsites doubles each).
export extern(C)
void manifold_get_psi(double* outRe, double* outIm) {
    int n4 = 4 * gLat.nsites;
    outRe[0 .. n4] = gLat.psiRe[0 .. n4];
    outIm[0 .. n4] = gLat.psiIm[0 .. n4];
}

/// Compute total norm² of the wavefunction.
export extern(C)
double manifold_norm2() {
    double n2 = 0;
    int n4 = 4 * gLat.nsites;
    foreach (i; 0 .. n4)
        n2 += gLat.psiRe[i]*gLat.psiRe[i] + gLat.psiIm[i]*gLat.psiIm[i];
    return n2;
}

/// Compute per-site probability |ψ_s|² for each site, write to outProbs.
/// Also returns Σ|ψ_s|⁴ (the IPR = inverse participation ratio denominator).
export extern(C)
double manifold_site_probs(double* outProbs) {
    int ns = gLat.nsites;
    double ipr = 0;
    foreach (s; 0 .. ns) {
        double p = 0;
        foreach (a; 0 .. 4) {
            double re = gLat.psiRe[4*s+a];
            double im = gLat.psiIm[4*s+a];
            p += re*re + im*im;
        }
        outProbs[s] = p;
        ipr += p * p;
    }
    return ipr;
}

/// Compute return probability |⟨ψ(t)|ψ(0)⟩|² for initial state at site 0.
/// Assumes IC was (1,0,i,0)/√2 at site 0.
export extern(C)
double manifold_return_prob() {
    // ⟨ψ₀|ψ(t)⟩ where ψ₀ = (1,0,i,0)/√2
    double re = gLat.psiRe[0] / SQRT2 + gLat.psiIm[2] / SQRT2;
    double im = gLat.psiIm[0] / SQRT2 - gLat.psiRe[2] / SQRT2;
    return re*re + im*im;
}
private enum SQRT2 = 1.4142135623730951;

// ---- Walk step ----

/// Run one full walk step: S_L → Vmix_L → S_R → Vmix_R.
/// Returns norm² after the step.
export extern(C)
double manifold_step(double mixPhi) {
    pureShift!false(*gLat, false);
    applyVmix!false(*gLat, false, mixPhi);
    pureShift!false(*gLat, true);
    applyVmix!false(*gLat, true, mixPhi);
    return manifold_norm2();
}

/// Run nSteps walk steps, recording observables at each step.
/// outNorms[t] = norm², outPR[t] = participation ratio, outReturn[t] = return prob.
export extern(C)
void manifold_run_observe(int nSteps, double mixPhi,
                          double* outNorms, double* outPR, double* outReturn) {
    int ns = gLat.nsites;
    auto probs = new double[ns];

    foreach (t; 0 .. nSteps) {
        double norm2 = manifold_step(mixPhi);
        double ipr = manifold_site_probs(probs.ptr);
        outNorms[t] = norm2;
        outPR[t] = (norm2 * norm2) / ipr;  // PR = (Σp)² / Σp²
        outReturn[t] = manifold_return_prob();
    }
}

// ===========================================================================
//  Manifold lattice builder — trace chains through face adjacency in D
// ===========================================================================

/// Standard tetrahedral exit directions (unit vectors toward each vertex).
private immutable double[3][4] STD_DIRS = () {
    import std.math : sqrt;
    double[3][4] d;
    d[0] = [0.0, 0.0, 1.0];
    d[1] = [2*sqrt(2.0)/3, 0.0, -1.0/3];
    d[2] = [-sqrt(2.0)/3, sqrt(6.0)/3, -1.0/3];
    d[3] = [-sqrt(2.0)/3, -sqrt(6.0)/3, -1.0/3];
    return d;
}();

private immutable int[4] L_PERM = [1, 3, 0, 2];
private immutable int[4] INV_L_PERM = [2, 0, 3, 1];

/// Triangulation data (set by manifold_load_triangulation).
private struct Triangulation {
    int nTets;
    int[][] tets;       // tets[i] = sorted 4-tuple of vertex labels
    int[][] neighbors;  // neighbors[i][f] = neighbor tet across face f
}
private Triangulation gTri;

/// Load triangulation from arrays passed by Python.
/// tets: nTets*4 ints (row-major, each row is 4 sorted vertex labels)
/// neighbors: nTets*4 ints (row-major, neighbors[i][f] = neighbor across face f)
export extern(C)
void manifold_load_triangulation(const(int)* tets, const(int)* neighbors, int nTets) {
    gTri.nTets = nTets;
    gTri.tets = new int[][nTets];
    gTri.neighbors = new int[][nTets];
    foreach (i; 0 .. nTets) {
        gTri.tets[i] = new int[4];
        gTri.neighbors[i] = new int[4];
        foreach (f; 0 .. 4) {
            gTri.tets[i][f] = tets[4*i + f];
            gTri.neighbors[i][f] = neighbors[4*i + f];
        }
    }
}

/// A manifold site state: (tet_id, vertex_perm[0..4]).
/// The vertex perm maps window positions to manifold vertex labels.
private struct MState {
    int tetId;
    int[4] perm;

    size_t toHash() const nothrow @safe {
        // Simple hash combining tet and perm
        size_t h = cast(size_t) tetId * 2654435761;
        foreach (p; perm)
            h ^= cast(size_t) p * 2246822519 + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
    bool opEquals(ref const MState o) const nothrow @safe {
        return tetId == o.tetId && perm == o.perm;
    }
}

/// Advance one step along a chain: drop perm[0], cross face, pick up new vertex.
private MState chainStep(MState s) {
    int dropped = s.perm[0];
    int[] tetVerts = gTri.tets[s.tetId];
    // Find which face is opposite the dropped vertex
    int faceIdx = -1;
    foreach (f; 0 .. 4)
        if (tetVerts[f] == dropped) { faceIdx = f; break; }
    assert(faceIdx >= 0);

    int neighborId = gTri.neighbors[s.tetId][faceIdx];
    int[] nbVerts = gTri.tets[neighborId];

    // Find the new vertex (in neighbor but not in shared face)
    int newVert = -1;
    foreach (v; nbVerts) {
        bool inOld = false;
        foreach (p; s.perm[1..4])
            if (v == p) { inOld = true; break; }
        if (!inOld) { newVert = v; break; }
    }
    assert(newVert >= 0);

    MState next;
    next.tetId = neighborId;
    next.perm = [s.perm[1], s.perm[2], s.perm[3], newVert];
    return next;
}

/// Trace a chain from a starting state until it loops back.
/// Returns the chain as an array of MStates.
private MState[] traceChain(MState start) {
    MState[] chain;
    chain.reserve(1024);
    MState s = start;
    do {
        chain ~= s;
        s = chainStep(s);
    } while (s != start);
    return chain;
}

/// Get the exit direction for a state: STD_DIRS[k] where k is the
/// position of perm[0] in the sorted tet vertex list.
private Vec3 exitDir(MState s) {
    int[] tetVerts = gTri.tets[s.tetId];
    foreach (k; 0 .. 4)
        if (tetVerts[k] == s.perm[0])
            return Vec3(STD_DIRS[k][0], STD_DIRS[k][1], STD_DIRS[k][2]);
    assert(false, "perm[0] not found in tet vertices");
}

/// Convert R-state to L-state at the same site.
private MState rToL(MState r) {
    MState l;
    l.tetId = r.tetId;
    foreach (k; 0 .. 4)
        l.perm[k] = r.perm[L_PERM[k]];
    return l;
}

/// Convert L-state to R-state at the same site.
private MState lToR(MState l) {
    MState r;
    r.tetId = l.tetId;
    foreach (k; 0 .. 4)
        r.perm[k] = l.perm[INV_L_PERM[k]];
    return r;
}

/// Build the full manifold lattice: trace all R and L chains,
/// create sites and closed chains in the global lattice.
///
/// Returns: number of sites created.
export extern(C)
int manifold_build_lattice() {
    import std.stdio : stderr;

    // Site registry: R-state → site ID
    int[MState] siteOf;
    // Exit directions per site per chirality
    Vec3[] rExitDirs;   // indexed by site ID
    Vec3[] lExitDirs;

    int nextSid = 0;

    int getSite(MState rState) {
        auto p = rState in siteOf;
        if (p) return *p;
        int sid = nextSid++;
        siteOf[rState] = sid;
        // Grow exit dir arrays
        if (rExitDirs.length <= sid) {
            rExitDirs.length = sid + 1;
            lExitDirs.length = sid + 1;
        }
        return sid;
    }

    // BFS over R-chain orbits
    bool[MState] visitedR;
    bool[MState] visitedL;
    MState[] rQueue;

    // Collected chains: (siteIds[], exitDirs[], isR)
    struct ChainData {
        int[] sids;
        Vec3[] dirs;
        bool isR;
    }
    ChainData[] allChains;

    // Seed with tet 0
    MState startR;
    startR.tetId = 0;
    foreach (k; 0 .. 4)
        startR.perm[k] = gTri.tets[0][k];
    rQueue ~= startR;

    while (rQueue.length > 0) {
        MState rs = rQueue[$ - 1];
        rQueue.length--;
        if (rs in visitedR) continue;

        // Trace R-chain
        MState[] rChain = traceChain(rs);
        foreach (ref st; rChain)
            visitedR[st] = true;

        // Assign sites and collect R exit dirs
        ChainData rcd;
        rcd.isR = true;
        rcd.sids = new int[rChain.length];
        rcd.dirs = new Vec3[rChain.length];
        foreach (i, ref st; rChain) {
            int sid = getSite(st);
            rcd.sids[i] = sid;
            rcd.dirs[i] = exitDir(st);
            rExitDirs[sid] = rcd.dirs[i];
        }
        allChains ~= rcd;

        stderr.writefln("  R-chain %d: period=%d, sites=%d",
                        cast(int) allChains.length - 1, rChain.length, nextSid);

        // Trace L-chains at each R-chain site
        foreach (ref st; rChain) {
            MState lStart = rToL(st);
            if (lStart in visitedL) continue;

            MState[] lChain = traceChain(lStart);
            foreach (ref ls; lChain)
                visitedL[ls] = true;

            ChainData lcd;
            lcd.isR = false;
            lcd.sids = new int[lChain.length];
            lcd.dirs = new Vec3[lChain.length];
            foreach (i, ref ls; lChain) {
                MState rEquiv = lToR(ls);
                int sid = getSite(rEquiv);
                lcd.sids[i] = sid;
                lcd.dirs[i] = exitDir(ls);
                lExitDirs[sid] = lcd.dirs[i];

                // Queue new R-states
                if (rEquiv !in visitedR)
                    rQueue ~= rEquiv;
            }
            allChains ~= lcd;
        }
    }

    stderr.writefln("  Chain tracing done: %d sites, %d chains", nextSid, allChains.length);

    // Create the D lattice
    gLat = new Lattice!false;
    *gLat = Lattice!false.create(nextSid + 100);
    foreach (i; 0 .. nextSid)
        gLat.allocSite(Vec3(0, 0, 0));

    // Create closed chains
    foreach (ref cd; allChains) {
        int n = cast(int) cd.sids.length;
        int chainId = cast(int) gLat.chains.length;
        Lattice!false.ChainT newChain;
        newChain.isR = cd.isR;
        newChain.isClosed = true;
        newChain.rootSite = cd.sids[0];
        newChain.rootIdx = 0;

        alias Ops = Lattice!false.Ops;
        foreach (i; 0 .. n) {
            int sid = cd.sids[i];
            Mat4 tau = makeTau(cd.dirs[i]);
            int pi = (i > 0) ? i - 1 : n - 1;
            int ni = (i < n - 1) ? i + 1 : 0;
            Mat4 tauP = makeTau(cd.dirs[pi]);
            Mat4 tauN = makeTau(cd.dirs[ni]);

            Ops op;
            op.siteId = sid;
            op.fwdBlock = mul(frameTransport(tau, tauN), projPlus(tau));
            op.bwdBlock = mul(frameTransport(tau, tauP), projMinus(tau));
            newChain.ops.pushBack(op);

            if (cd.isR) { gLat.sites[sid].rChain = chainId; gLat.sites[sid].rIdx = i; }
            else        { gLat.sites[sid].lChain = chainId; gLat.sites[sid].lIdx = i; }
        }
        gLat.chains ~= newChain;
    }

    stderr.writefln("  Lattice build done: %d sites, %d chains",
                    gLat.nsites, gLat.chains.length);

    return nextSid;
}
