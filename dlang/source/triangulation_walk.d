/**
 * triangulation_walk.d — Walker paths on a triangulated manifold.
 *
 * Builds a SiteContainer from a triangulation by tracing BC helix chains
 * through face-adjacent tetrahedra. Each unique manifold state (tetId, perm)
 * gets exactly one site. Chains are open in the site container; the manifold
 * topology is encoded in a redirect table that tells the walk operator
 * where to send chain-end overflow.
 *
 * This module knows about triangulation topology and the site container,
 * but nothing about spinors or walk operators.
 */
module triangulation_walk;

import site_container : SiteContainer, GeoChain, IS_R, IS_L;
import geometry : Vec3, chainCentroid, chainVertexDirs, chainExitDir;

/// A manifold walker state: (tet_id, vertex_perm[0..4]).
struct MState {
    int tetId;
    int[4] perm;

    size_t toHash() const nothrow @safe {
        size_t h = cast(size_t) tetId * 2654435761;
        foreach (p; perm)
            h ^= cast(size_t) p * 2246822519 + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
    bool opEquals(ref const MState o) const nothrow @safe {
        return tetId == o.tetId && perm == o.perm;
    }
}

/// Redirect: when overflow exits a chain end, send it to targetSite.
struct Redirect {
    int chainId;        /// chain in the site container
    bool isFwd;         /// true = forward end, false = backward end
    int targetSite;     /// site to receive the overflow
}

/// The triangulation data (loaded from Python).
struct Triangulation {
    int nTets;
    int[][] tets;       /// tets[i] = 4 sorted vertex labels
    int[][] neighbors;  /// neighbors[i][f] = neighbor tet across face f
}

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

/// Result of building the lattice from a triangulation.
struct TriangulationWalk {
    SiteContainer sites;
    Redirect[] redirects;
    int[] siteTet;          /// site ID → manifold tet ID
    int[MState] siteOf;     /// manifold R-state → site ID (for lookups)

    /// Look up all sites at a given tet.
    int[] sitesAtTet(int tetId) const {
        int[] result;
        foreach (sid; 0 .. sites.nsites)
            if (siteTet[sid] == tetId)
                result ~= sid;
        return result;
    }
}

// ---- Manifold chain tracing (private helpers) ----

/// Advance one step along a chain: drop perm[0], cross face, pick up new vertex.
private MState chainStep(ref const Triangulation tri, MState s) {
    int dropped = s.perm[0];
    auto tetVerts = tri.tets[s.tetId];
    int faceIdx = -1;
    foreach (f; 0 .. 4)
        if (tetVerts[f] == dropped) { faceIdx = f; break; }
    assert(faceIdx >= 0);

    int neighborId = tri.neighbors[s.tetId][faceIdx];
    auto nbVerts = tri.tets[neighborId];

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

/// Trace a chain until it loops back. Returns the full cycle of states.
private MState[] traceChain(ref const Triangulation tri, MState start) {
    MState[] chain;
    chain.reserve(64);
    MState s = start;
    do {
        chain ~= s;
        s = chainStep(tri, s);
    } while (s != start);
    return chain;
}

/// Get exit direction for a state.
private Vec3 exitDir(ref const Triangulation tri, MState s) {
    auto tetVerts = tri.tets[s.tetId];
    foreach (k; 0 .. 4)
        if (tetVerts[k] == s.perm[0])
            return Vec3(STD_DIRS[k][0], STD_DIRS[k][1], STD_DIRS[k][2]);
    assert(false, "perm[0] not found in tet vertices");
}

/// Get all 4 vertex directions for a state.
private Vec3[4] vertexDirs(ref const Triangulation tri, MState s) {
    auto tetVerts = tri.tets[s.tetId];
    Vec3[4] dirs;
    foreach (w; 0 .. 4)
        foreach (k; 0 .. 4)
            if (tetVerts[k] == s.perm[w]) {
                dirs[w] = Vec3(STD_DIRS[k][0], STD_DIRS[k][1], STD_DIRS[k][2]);
                break;
            }
    return dirs;
}

/// Convert R-state to L-state at the same site.
private MState rToL(MState r) {
    MState l;
    l.tetId = r.tetId;
    foreach (k; 0 .. 4) l.perm[k] = r.perm[L_PERM[k]];
    return l;
}

/// Convert L-state to R-state at the same site.
private MState lToR(MState l) {
    MState r;
    r.tetId = l.tetId;
    foreach (k; 0 .. 4) r.perm[k] = l.perm[INV_L_PERM[k]];
    return r;
}

/// Build a TriangulationWalk from a triangulation.
///
/// Algorithm:
/// 1. Trace all manifold chains to discover the topology.
/// 2. BFS-build the site container: for each manifold chain, grow from
///    an existing site, creating new sites via chainAppend. When a chain
///    step would revisit a manifold state already in the container,
///    record a redirect and keep extending (other states may be new).
/// 3. Every cross-chain stub gets extended into its full manifold chain.
TriangulationWalk buildFromTriangulation(ref const Triangulation tri) {
    import std.stdio : stderr;

    TriangulationWalk tw;

    // ---- Phase 1: Trace all manifold chains ----
    struct MChain {
        MState[] states;
        bool isR;
    }
    MChain[] mChains;

    bool[MState] visitedR, visitedL;
    MState[] rQueue;

    MState startR;
    startR.tetId = 0;
    foreach (k; 0 .. 4) startR.perm[k] = tri.tets[0][k];
    rQueue ~= startR;

    while (rQueue.length > 0) {
        MState rs = rQueue[$ - 1];
        rQueue.length--;
        if (rs in visitedR) continue;

        MState[] rStates = traceChain(tri, rs);
        foreach (ref st; rStates) visitedR[st] = true;
        mChains ~= MChain(rStates, true);

        foreach (ref st; rStates) {
            MState lStart = rToL(st);
            if (lStart in visitedL) continue;

            MState[] lStates = traceChain(tri, lStart);
            foreach (ref ls; lStates) visitedL[ls] = true;
            mChains ~= MChain(lStates, false);

            foreach (ref ls; lStates) {
                MState rEquiv = lToR(ls);
                if (rEquiv !in visitedR) rQueue ~= rEquiv;
            }
        }
    }

    stderr.writefln("  Traced %d manifold chains", mChains.length);

    // ---- Phase 2: Build site container ----

    // Map each manifold state (as R-state) to which R/L manifold chain it's on
    int[MState] rChainOf, lChainOf;
    foreach (ci, ref mc; mChains) {
        foreach (ref st; mc.states) {
            MState rState = mc.isR ? st : lToR(st);
            if (mc.isR) rChainOf[rState] = cast(int) ci;
            else        lChainOf[rState] = cast(int) ci;
        }
    }

    // Estimate total sites
    int totalStates = 0;
    foreach (ref mc; mChains) totalStates += cast(int) mc.states.length;

    tw.sites = SiteContainer.create(totalStates + 100);
    tw.siteTet = new int[totalStates + 100];
    tw.redirects.length = 0;

    bool[] chainBuilt = new bool[mChains.length];
    int[] buildQueue;

    // Helper: get the R-state for a manifold chain state
    MState toRState(ref const MChain mc, int i) {
        return mc.isR ? mc.states[i] : lToR(mc.states[i]);
    }

    // Helper: build one manifold chain into the site container.
    // Returns the lattice chain ID.
    int buildOneChain(int mci) {
        auto mc = &mChains[mci];
        int period = cast(int) mc.states.length;

        // Find entry point: which state already has a site?
        int entryIdx = -1;
        int rootSite = -1;
        foreach (j; 0 .. period) {
            MState rSt = toRState(*mc, j);
            auto p = rSt in tw.siteOf;
            if (p) { entryIdx = j; rootSite = *p; break; }
        }
        if (entryIdx < 0) return -1;

        // Create chain from entry point
        Vec3[4] vdirs = vertexDirs(tri, mc.states[entryIdx]);
        Vec3 c0 = tw.sites.sites[rootSite].pos;
        Vec3[4] rootVerts;
        foreach (k; 0 .. 4)
            rootVerts[k] = Vec3(c0.x + vdirs[k].x, c0.y + vdirs[k].y, c0.z + vdirs[k].z);

        int chainId = tw.sites.makeChain(rootSite, mc.isR, rootVerts);

        // Extend forward through the cycle
        foreach (j; 1 .. period) {
            int i = (entryIdx + j) % period;
            MState rSt = toRState(*mc, i);

            auto pExisting = rSt in tw.siteOf;
            if (pExisting) {
                // This state already has a site — record redirect
                Redirect rd;
                rd.chainId = chainId;
                rd.isFwd = true;
                rd.targetSite = *pExisting;
                tw.redirects ~= rd;

                // But DON'T break — we can't extend past a redirect because
                // the chain position formula needs continuity.
                // Instead, start a NEW chain segment from the existing site.
                // Actually: we must stop this chain here. The remaining
                // states will be covered when we build the chain starting
                // from the existing site's cross-chain.
                break;
            }

            // New site
            int newSite = tw.sites.chainAppend(chainId);
            tw.siteTet[newSite] = mc.states[i].tetId;
            tw.siteOf[rSt] = newSite;

            // Create cross-chain stub
            tw.sites.makeCrossChain(newSite, !mc.isR);

            // Queue the cross manifold chain
            MState rSt_i = toRState(*mc, i);
            auto pCross = mc.isR ? (rSt_i in lChainOf) : (rSt_i in rChainOf);
            if (pCross && !chainBuilt[*pCross])
                buildQueue ~= *pCross;
        }

        // Backward redirect: the state before the entry point
        int bwdIdx = (entryIdx - 1 + period) % period;
        MState bwdRSt = toRState(*mc, bwdIdx);
        auto pBwd = bwdRSt in tw.siteOf;
        if (pBwd) {
            Redirect rd;
            rd.chainId = chainId;
            rd.isFwd = false;
            rd.targetSite = *pBwd;
            tw.redirects ~= rd;
        }

        return chainId;
    }

    // Seed: create origin site, build first R-chain
    int originSite = tw.sites.allocSite(Vec3(0, 0, 0));
    MState originR = mChains[0].states[0];
    tw.siteOf[originR] = originSite;
    tw.siteTet[originSite] = originR.tetId;

    buildQueue ~= 0;

    // Iterate until all reachable chains are built
    bool progress = true;
    while (progress) {
        progress = false;

        // Process queued chains
        while (buildQueue.length > 0) {
            int mci = buildQueue[$ - 1];
            buildQueue.length--;
            if (chainBuilt[mci]) continue;
            chainBuilt[mci] = true;
            progress = true;

            buildOneChain(mci);
        }

        // Check for unbuilt chains that now have reachable entry points
        foreach (ci, ref mc; mChains) {
            if (chainBuilt[ci]) continue;
            foreach (ref st; mc.states) {
                MState rSt = mc.isR ? st : lToR(st);
                if (rSt in tw.siteOf) {
                    buildQueue ~= cast(int) ci;
                    break;
                }
            }
        }

        if (buildQueue.length > 0) progress = true;
    }

    int nBuilt = 0;
    foreach (b; chainBuilt) if (b) nBuilt++;
    stderr.writefln("  Built %d/%d chains, %d sites, %d redirects",
                    nBuilt, mChains.length, tw.sites.nsites, tw.redirects.length);

    return tw;
}
