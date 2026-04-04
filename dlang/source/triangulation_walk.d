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
/// Algorithm: true BFS growth from the origin site. At each step,
/// extend every chain that has an extendable end by one site.
/// When a chain extension would create a site at a manifold state
/// that already has a site, record a redirect instead.
/// Continue until all 7200 manifold states have sites.
TriangulationWalk buildFromTriangulation(ref const Triangulation tri) {
    import std.stdio : stderr;

    TriangulationWalk tw;

    // Count total manifold states for capacity estimate
    // (BFS traversal of manifold chains from tet 0)
    bool[MState] allR, allL;
    MState[] rq;
    MState startR;
    startR.tetId = 0;
    foreach (k; 0 .. 4) startR.perm[k] = tri.tets[0][k];
    rq ~= startR;
    while (rq.length > 0) {
        MState rs = rq[$ - 1]; rq.length--;
        if (rs in allR) continue;
        auto rStates = traceChain(tri, rs);
        foreach (ref st; rStates) allR[st] = true;
        foreach (ref st; rStates) {
            MState ls = rToL(st);
            if (ls in allL) continue;
            auto lStates = traceChain(tri, ls);
            foreach (ref l2; lStates) allL[l2] = true;
            foreach (ref l2; lStates) {
                MState re = lToR(l2);
                if (re !in allR) rq ~= re;
            }
        }
    }
    int totalStates = cast(int) allR.length;
    stderr.writefln("  Total manifold states: %d", totalStates);

    tw.sites = SiteContainer.create(totalStates + 100);
    tw.siteTet = new int[totalStates + 100];
    tw.redirects.length = 0;

    // ---- BFS growth ----
    // Each site in the tree is on one R-chain and one L-chain.
    // A "frontier" is a chain that can be extended (its manifold successor
    // state doesn't have a site yet, or needs a redirect recorded).
    //
    // We track, for each lattice chain, which manifold state comes next
    // if we extend it forward.

    // Per lattice chain: the manifold state sequence and current extension index.
    struct ChainInfo {
        MState[] mStates;   // full manifold cycle for this chain's chirality
        int entryIdx;       // which position in mStates is the chain root
        int extended;       // how many sites beyond root have been created
        bool isR;
    }
    ChainInfo[] chainInfos;

    /// Create the origin site and its two chains (R + L).
    int originSite = tw.sites.allocSite(Vec3(0, 0, 0));
    tw.siteOf[startR] = originSite;
    tw.siteTet[originSite] = startR.tetId;

    // Helper: get the R-state for a manifold state on a chain
    MState toRState(MState st, bool isR) {
        return isR ? st : lToR(st);
    }

    // Helper: trace the manifold chain containing a given state
    // and return (states[], index of the given state in the cycle).
    auto traceAndFind(MState st, bool isR) {
        auto states = traceChain(tri, st);
        int idx = -1;
        foreach (j, ref s; states)
            if (s == st) { idx = cast(int) j; break; }
        struct Result { MState[] states; int idx; }
        return Result(states, idx);
    }

    // Create origin's R-chain
    {
        auto mf = traceAndFind(startR, true);
        Vec3[4] vdirs = vertexDirs(tri, startR);
        Vec3[4] rootVerts;
        foreach (k; 0 .. 4)
            rootVerts[k] = Vec3(vdirs[k].x, vdirs[k].y, vdirs[k].z); // pos=(0,0,0)
        int cid = tw.sites.makeChain(originSite, true, rootVerts);
        chainInfos ~= ChainInfo(mf.states, mf.idx, 0, true);
    }
    // Create origin's L-chain
    {
        MState lStart = rToL(startR);
        auto mf = traceAndFind(lStart, false);
        Vec3[4] vdirs = vertexDirs(tri, lStart);
        Vec3[4] rootVerts;
        foreach (k; 0 .. 4)
            rootVerts[k] = Vec3(vdirs[k].x, vdirs[k].y, vdirs[k].z);
        int cid = tw.sites.makeChain(originSite, false, rootVerts);
        chainInfos ~= ChainInfo(mf.states, mf.idx, 0, false);
    }

    // BFS loop: extend every chain by one site, round-robin, until done.
    int nCreated = 1;  // origin
    int round = 0;
    while (nCreated < totalStates) {
        int createdThisRound = 0;
        int nChains = cast(int) chainInfos.length;

        foreach (ci; 0 .. nChains) {
            auto info = &chainInfos[ci];
            int period = cast(int) info.mStates.length;

            // How many more can we extend?
            if (info.extended >= period - 1) continue;  // fully extended

            // Next manifold state
            int nextJ = (info.entryIdx + info.extended + 1) % period;
            MState nextMState = info.mStates[nextJ];
            MState nextR = toRState(nextMState, info.isR);

            auto pExisting = nextR in tw.siteOf;
            if (pExisting) {
                // Redirect — this chain can't grow further
                Redirect rd;
                rd.chainId = ci;
                rd.isFwd = true;
                rd.targetSite = *pExisting;
                tw.redirects ~= rd;
                info.extended = period;  // mark as done
                continue;
            }

            // Create new site
            int newSite = tw.sites.chainAppend(ci);
            tw.siteTet[newSite] = nextMState.tetId;
            tw.siteOf[nextR] = newSite;
            info.extended++;
            nCreated++;
            createdThisRound++;

            // Create cross-chain at new site
            tw.sites.makeCrossChain(newSite, !info.isR);
            // The cross-chain stub is a new lattice chain — register it.
            int crossChainId = cast(int) tw.sites.chains.length - 1;

            // Find the manifold chain for this cross-chain
            MState crossMState = info.isR ? rToL(nextMState) : nextMState;
            bool crossIsR = !info.isR;
            // If cross is L, the manifold state is the L-state; trace from there
            // If cross is R, the manifold state is the R-state = nextR
            MState crossTrace = crossIsR ? nextR : rToL(info.mStates[nextJ]);
            // Actually for the cross chain: if we built an R-chain site,
            // its cross is an L-chain. The L-state at this site is rToL(R-state).
            if (info.isR) {
                auto mf = traceAndFind(rToL(info.mStates[nextJ]), false);
                chainInfos ~= ChainInfo(mf.states, mf.idx, 0, false);
            } else {
                // Built an L-chain site, cross is R-chain.
                // The R-state is lToR(L-state) = nextR (already computed above
                // for L-chains: nextR = lToR(nextMState))
                auto mf = traceAndFind(lToR(info.mStates[nextJ]), true);
                chainInfos ~= ChainInfo(mf.states, mf.idx, 0, true);
            }
        }

        round++;
        if (round % 10 == 0 || createdThisRound == 0)
            stderr.writefln("  Round %d: %d sites, %d chains, %d redirects",
                            round, nCreated, chainInfos.length, tw.redirects.length);

        if (createdThisRound == 0) break;  // no progress — all chains done or redirected
    }

    // ---- Add backward redirects for every chain ----
    // Each chain's backward end (before its root) connects to the previous
    // manifold state, which should have a site by now.
    foreach (ci_u, ref info; chainInfos) {
        int ci = cast(int) ci_u;
        int period = cast(int) info.mStates.length;
        int bwdIdx = (info.entryIdx - 1 + period) % period;
        MState bwdR = toRState(info.mStates[bwdIdx], info.isR);
        auto pBwd = bwdR in tw.siteOf;
        if (pBwd) {
            Redirect rd;
            rd.chainId = ci;
            rd.isFwd = false;
            rd.targetSite = *pBwd;
            tw.redirects ~= rd;
        }
    }

    stderr.writefln("  Build done: %d sites, %d chains, %d redirects",
                    tw.sites.nsites, tw.sites.chains.length, tw.redirects.length);

    return tw;
}
