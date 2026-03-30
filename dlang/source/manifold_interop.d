/**
 * manifold_interop.d — C interface for building and running a manifold walk.
 *
 * Python creates the lattice structure (sites, closed chains) via these
 * functions, then runs the walk loop in D for performance.
 */
module manifold_interop;

import geometry : Vec3, ChainOrigin, computeChainOrigin, chainCentroid;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, matVecSplit;
import lattice : Lattice;
import operators : pureShift, applyVmix, initWavepacket;

/// Opaque handle for the lattice.  We store a heap-allocated pointer
/// so the GC keeps it alive across C calls.
private Lattice!false* gLat;
private int[] gSiteTet;   // site_id → manifold tet_id (set by manifold_build_lattice)

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

/// Copy site→tet mapping into caller buffer. Buffer must hold nsites ints.
export extern(C)
void manifold_get_site_tets(int* outTets) {
    int ns = gLat.nsites;
    outTets[0 .. ns] = gSiteTet[0 .. ns];
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

/// Initialize a Gaussian wavepacket using the shared initWavepacket.
/// BFS frame-transports the balanced spinor (1,0,i,0)/√2 from originSite,
/// applies Gaussian envelope centered at originSite and momentum kick.
export extern(C)
void manifold_init_wavepacket(int originSite, double sigma,
                              double kx, double ky, double kz) {
    initWavepacket!false(*gLat, originSite, sigma, kx, ky, kz);
}

/// Get site positions (3 doubles per site: x, y, z).
export extern(C)
void manifold_get_site_positions(double* outPos) {
    int ns = gLat.nsites;
    foreach (s; 0 .. ns) {
        outPos[3*s]   = gLat.sites[s].pos.x;
        outPos[3*s+1] = gLat.sites[s].pos.y;
        outPos[3*s+2] = gLat.sites[s].pos.z;
    }
}

/// Compute ⟨x⟩, ⟨y⟩, ⟨z⟩, ⟨r²⟩ weighted by |ψ|².
export extern(C)
void manifold_mean_position(double* out4) {
    int ns = gLat.nsites;
    double sumP = 0, sumX = 0, sumY = 0, sumZ = 0, sumR2 = 0;
    foreach (s; 0 .. ns) {
        double prob = 0;
        foreach (a; 0 .. 4) {
            double re = gLat.psiRe[4*s+a];
            double im = gLat.psiIm[4*s+a];
            prob += re*re + im*im;
        }
        Vec3 p = gLat.sites[s].pos;
        sumP  += prob;
        sumX  += prob * p.x;
        sumY  += prob * p.y;
        sumZ  += prob * p.z;
        sumR2 += prob * (p.x*p.x + p.y*p.y + p.z*p.z);
    }
    double inv = 1.0 / sumP;
    out4[0] = sumX * inv;
    out4[1] = sumY * inv;
    out4[2] = sumZ * inv;
    out4[3] = sumR2 * inv;
}

/// Run a wavepacket kick test in 3D. Init from originSite with frame-transported
/// balanced spinor, Gaussian envelope, and momentum kick (kx,ky,kz).
/// Run nSteps, record ⟨x⟩,⟨y⟩,⟨z⟩,⟨r²⟩ at each step.
/// out: 4*(nSteps+1) doubles, row-major [t][xyzr2].
export extern(C)
void manifold_kick_run_3d(int originSite,
                          double sigma, double kx, double ky, double kz,
                          double mixPhi, int nSteps, double* out4) {
    manifold_init_wavepacket(originSite, sigma, kx, ky, kz);

    double[4] obs = 0;
    manifold_mean_position(obs.ptr);
    out4[0..4] = obs[];

    foreach (t; 1 .. nSteps + 1) {
        manifold_step(mixPhi);
        manifold_mean_position(obs.ptr);
        out4[4*t .. 4*t+4] = obs[];
    }
}

/// Compute mean and variance of a per-site observable weighted by |ψ|².
/// coords[s] = observable value at site s.
/// Returns: outMean[0] = ⟨x⟩, outMean[1] = ⟨x²⟩ (for computing spread).
export extern(C)
void manifold_mean_coord(const(double)* coords, double* outMean) {
    int ns = gLat.nsites;
    double sumP = 0, sumPX = 0, sumPX2 = 0;
    foreach (s; 0 .. ns) {
        double p = 0;
        foreach (a; 0 .. 4) {
            double re = gLat.psiRe[4*s+a];
            double im = gLat.psiIm[4*s+a];
            p += re*re + im*im;
        }
        double x = coords[s];
        sumP += p;
        sumPX += p * x;
        sumPX2 += p * x * x;
    }
    outMean[0] = sumPX / sumP;    // ⟨d⟩
    outMean[1] = sumPX2 / sumP;   // ⟨d²⟩
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

/// Get all 4 vertex directions for a state.
/// dirs[window_pos] = STD_DIRS[tet_pos] where tet_pos is the sorted
/// position of perm[window_pos] in the tet vertex list.
private Vec3[4] vertexDirs(MState s) {
    int[] tetVerts = gTri.tets[s.tetId];
    Vec3[4] dirs;
    foreach (w; 0 .. 4) {
        foreach (k; 0 .. 4) {
            if (tetVerts[k] == s.perm[w]) {
                dirs[w] = Vec3(STD_DIRS[k][0], STD_DIRS[k][1], STD_DIRS[k][2]);
                break;
            }
        }
    }
    return dirs;
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
    Vec3[] rExitDirs;   // indexed by site ID
    Vec3[] lExitDirs;

    int nextSid = 0;

    int getSite(MState rState) {
        auto p = rState in siteOf;
        if (p) return *p;
        int sid = nextSid++;
        siteOf[rState] = sid;
        if (rExitDirs.length <= sid) {
            rExitDirs.length = sid + 1;
            lExitDirs.length = sid + 1;
        }
        if (gSiteTet.length <= sid)
            gSiteTet.length = sid + 1;
        gSiteTet[sid] = rState.tetId;
        return sid;
    }

    // BFS over R-chain orbits
    bool[MState] visitedR;
    bool[MState] visitedL;
    MState[] rQueue;

    // Collected chains: (siteIds[], exitDirs[], isR, origin for position computation)
    struct ChainData {
        int[] sids;
        Vec3[] dirs;
        bool isR;
        ChainOrigin origin;
    }
    ChainData[] allChains;

    // Position storage: set when a site is first encountered
    Vec3[] sitePositions;
    bool[] positionSet;

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

        // Assign sites, collect R exit dirs, compute positions
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

        // Compute ChainOrigin for this R-chain using the first site's geometry.
        // If the first site already has a position (from another chain), use it;
        // otherwise start at origin.
        {
            int firstSid = rcd.sids[0];
            if (sitePositions.length <= firstSid) {
                sitePositions.length = firstSid + 1;
                positionSet.length = firstSid + 1;
            }
            Vec3 startPos = positionSet[firstSid] ? sitePositions[firstSid] : Vec3(0, 0, 0);
            Vec3[4] vdirs = vertexDirs(rChain[0]);
            // face = 0: exit direction is always at window position 0
            rcd.origin = computeChainOrigin(startPos, vdirs, 0, true);

            // Set positions for all sites on this chain
            foreach (i; 0 .. rChain.length) {
                int sid = rcd.sids[i];
                if (sitePositions.length <= sid) {
                    sitePositions.length = sid + 1;
                    positionSet.length = sid + 1;
                }
                if (!positionSet[sid]) {
                    sitePositions[sid] = chainCentroid(&rcd.origin, cast(int) i);
                    positionSet[sid] = true;
                }
            }
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

                if (rEquiv !in visitedR)
                    rQueue ~= rEquiv;
            }

            // Compute L-chain origin and positions
            {
                int firstSid = lcd.sids[0];
                if (sitePositions.length <= firstSid) {
                    sitePositions.length = firstSid + 1;
                    positionSet.length = firstSid + 1;
                }
                Vec3 startPos = positionSet[firstSid] ? sitePositions[firstSid] : Vec3(0, 0, 0);
                Vec3[4] vdirs = vertexDirs(lChain[0]);
                lcd.origin = computeChainOrigin(startPos, vdirs, 0, false);

                foreach (i; 0 .. lChain.length) {
                    int sid = lcd.sids[i];
                    if (sitePositions.length <= sid) {
                        sitePositions.length = sid + 1;
                        positionSet.length = sid + 1;
                    }
                    if (!positionSet[sid]) {
                        sitePositions[sid] = chainCentroid(&lcd.origin, cast(int) i);
                        positionSet[sid] = true;
                    }
                }
            }

            allChains ~= lcd;
        }
    }

    stderr.writefln("  Chain tracing done: %d sites, %d chains", nextSid, allChains.length);

    // Create the D lattice with computed positions
    gLat = new Lattice!false;
    *gLat = Lattice!false.create(nextSid + 100);
    foreach (i; 0 .. nextSid)
        gLat.allocSite(sitePositions[i]);

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
