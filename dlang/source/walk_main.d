/**
 * walk_main.d — Main entry point for the 3D adaptive quantum walk.
 *
 * Uses Lattice!false (no precomputed coin) for the V-mixing-only walk
 * (theta=0). The coin step is compiled out entirely.
 */
module walk_main;

import std.math : sqrt, cos, sin, exp;
import std.conv : to;
import std.stdio : writef, writefln, stderr, stdout;
import geometry : Vec3, dot, STEP_LEN;
import lattice : Lattice, ProximityGrid, generateSites, IS_R, IS_L, isRamLow, freeRamBytes;
import dirac : Mat4, makeTau, frameTransport, matVecSplit, projPlus, projMinus;
import operators : applyShift, applyCoin, applyVmix, ShiftResult;
import observables : computeObservables, Observables;
import symmetry : checkA4Symmetry;

enum bool HAS_COIN = false;

struct K0Vec { double x = 0, y = 0, z = 0; }

struct WalkParams {
    double theta = 0.0;        // default: no coin (V-mixing only)
    double sigma = 3.0;
    int nSteps = 50;
    double threshold = 1e-10;  // extension threshold (absorb if amp < this)
    double pruneThresh = 1e-6; // pruning threshold (prune chain ends below this)
    double mixPhi = 0.0;
    double seedThresh = 1e-4;  // amplitude cutoff for site generation
    double dMin = 0.35;        // min distance between sites (0 = no limit)
    K0Vec[] kicks;             // momentum kick vectors
}

/// Parse args: theta sigma nsteps threshold pruneThresh mixPhi seedThresh dMin [kicks]
WalkParams parseArgs(string[] args) {
    import std.array : split;
    WalkParams p;
    if (args.length > 1) p.theta = args[1].to!double;
    if (args.length > 2) p.sigma = args[2].to!double;
    if (args.length > 3) p.nSteps = args[3].to!int;
    if (args.length > 4) p.threshold = args[4].to!double;
    if (args.length > 5) p.pruneThresh = args[5].to!double;
    if (args.length > 6) p.mixPhi = args[6].to!double;
    if (args.length > 7) p.seedThresh = args[7].to!double;
    if (args.length > 8) p.dMin = args[8].to!double;
    if (args.length > 9) {
        foreach (triple; args[9].split(";")) {
            auto c = triple.split(",");
            K0Vec k;
            if (c.length > 0) k.x = c[0].to!double;
            if (c.length > 1) k.y = c[1].to!double;
            if (c.length > 2) k.z = c[2].to!double;
            p.kicks ~= k;
        }
    }
    if (p.kicks.length == 0) p.kicks = [K0Vec(0, 0, 0)];
    return p;
}

/// Compute frame-transport tau for site s on chain type isR.
private Mat4 siteTau(ref Lattice!HAS_COIN lat, int s, bool isR) {
    return makeTau(lat.exitDirForSite(s, isR));
}

void initWavepacket(ref Lattice!HAS_COIN lat, double sigma,
                    double k0x, double k0y, double k0z) {
    int ns = lat.nsites;

    // BFS to compute frame-transported reference spinor at each site.
    // refRe/refIm[4*n..4*n+4] holds U(origin→n) · (1,0,0,0).
    auto refRe = new double[4 * ns];  refRe[] = 0;
    auto refIm = new double[4 * ns];  refIm[] = 0;
    auto visited = new bool[ns];      visited[] = false;

    // Build fully balanced reference spinor at the origin.
    // Find ψ with <β>=0 and <α_i>=0 for all i, which gives <τ_a>=0 for all 4
    // tetrahedral faces. This ensures zero net current in every direction.
    //
    // Method: β and α_i are 4 Hermitian matrices. We need ψ in the intersection
    // of their zero-expectation surfaces. Since τ = νβ + (3/4)(d·α), the
    // conditions <β>=0 and <α_i>=0 automatically give <τ_a>=0 for all a.
    //
    // <β>=0 means |ψ₀|²+|ψ₁|² = |ψ₂|²+|ψ₃|² (equal upper/lower weight).
    // <α_i>=0 gives 3 more real constraints. With 8 real parameters (4 complex)
    // minus normalization and global phase = 6 DOF, we have 6-4=2 free parameters.
    {
        // Dirac matrices (matching dirac.d)
        // β = diag(1,1,-1,-1)
        // α₁: (0,3)↔1, (1,2)↔1
        // α₂: (0,3)↔-i, (1,2)↔i
        // α₃: (0,2)↔1, (1,3)↔-1

        // Represent ψ = (a, b, c, d) with a,b,c,d complex.
        // <β> = |a|²+|b|²-|c|²-|d|² = 0  →  |a|²+|b|² = |c|²+|d|² = 1/2
        // <α₁> = 2Re(a*conj(d) + b*conj(c)) = 0
        // <α₂> = 2Re(i*a*conj(d) - i*b*conj(c)) = 2Im(-a*conj(d) + b*conj(c)) = 0
        // <α₃> = 2Re(a*conj(c) - b*conj(d)) = 0
        //
        // A simple solution: set b=d=0, then <β>=0 requires |a|=|c|.
        // <α₁> = 0 (both terms zero since b=d=0).
        // <α₂> = 0 (same).
        // <α₃> = 2Re(a*conj(c)) = 0, so a and c must be 90° out of phase.
        // Solution: a = 1/√2, c = i/√2.

        refRe[0] = 1.0 / sqrt(2.0);  refIm[0] = 0;
        refRe[1] = 0;                 refIm[1] = 0;
        refRe[2] = 0;                 refIm[2] = 1.0 / sqrt(2.0);
        refRe[3] = 0;                 refIm[3] = 0;

        // Verify balance
        double expBeta = refRe[0]*refRe[0] + refIm[0]*refIm[0]
                       + refRe[1]*refRe[1] + refIm[1]*refIm[1]
                       - refRe[2]*refRe[2] - refIm[2]*refIm[2]
                       - refRe[3]*refRe[3] - refIm[3]*refIm[3];
        stderr.writefln("  IC: fully balanced (1,0,i,0)/√2  <β>=%.6f", expBeta);
    }
    visited[0] = true;

    // BFS queue
    int[] queue;
    queue.reserve(ns);
    queue ~= 0;
    int qHead = 0;

    while (qHead < queue.length) {
        int s = queue[qHead++];

        // Try all 4 chain neighbors: R-next, R-prev, L-next, L-prev
        static immutable bool[2] chiralities = [true, false];
        foreach (isR; chiralities) {
            if (!lat.hasChain(s, isR)) continue;
            Mat4 tauS = siteTau(lat, s, isR);

            foreach (dir; 0 .. 2) {  // 0=next, 1=prev
                int nb = (dir == 0) ? lat.chainNext(s, isR) : lat.chainPrev(s, isR);
                if (nb < 0 || visited[nb]) continue;

                // Compute U(s→nb) and apply to ref spinor at s
                Mat4 tauNb = siteTau(lat, nb, isR);
                Mat4 U = frameTransport(tauS, tauNb);

                double[4] outRe = 0, outIm = 0;
                matVecSplit(U, &refRe[4*s], &refIm[4*s], outRe.ptr, outIm.ptr);
                refRe[4*nb .. 4*nb+4] = outRe[];
                refIm[4*nb .. 4*nb+4] = outIm[];

                visited[nb] = true;
                queue ~= nb;
            }
        }
    }

    int nVisited = 0;
    foreach (v; visited[0 .. ns]) if (v) nVisited++;
    stderr.writefln("  IC transport: %d/%d sites reached from origin", nVisited, ns);

    // Apply Gaussian envelope and momentum kick to transported spinors
    double norm2 = 0;
    foreach (n; 0 .. ns) {
        double r2 = dot(lat.sites[n].pos, lat.sites[n].pos);
        double w = exp(-r2 / (2 * sigma * sigma));
        double phase = k0x * lat.sites[n].pos.x
                     + k0y * lat.sites[n].pos.y
                     + k0z * lat.sites[n].pos.z;
        double phRe = cos(phase);
        double phIm = sin(phase);

        // psi = w * e^{ik·r} * ref_spinor  (complex multiply)
        foreach (a; 0 .. 4) {
            double rr = refRe[4*n + a], ri = refIm[4*n + a];
            lat.psiRe[4*n + a] = w * (phRe * rr - phIm * ri);
            lat.psiIm[4*n + a] = w * (phRe * ri + phIm * rr);
        }
        if (visited[n])
            norm2 += w * w;  // only count sites with nonzero spinor
    }
    double nf = 1.0 / sqrt(norm2);
    foreach (i; 0 .. 4 * ns) {
        lat.psiRe[i] *= nf;
        lat.psiIm[i] *= nf;
    }
}

void run(WalkParams p) {
    double extThresh2 = p.threshold * p.threshold;
    double pruneThresh2 = p.pruneThresh * p.pruneThresh;

    stderr.writefln("=== walk_d: theta=%.3f sigma=%.1f steps=%d ext_thresh=%.1e prune_thresh=%.1e mix_phi=%.4f kicks=%d ===",
                    p.theta, p.sigma, p.nSteps, p.threshold, p.pruneThresh, p.mixPhi,
                    cast(int) p.kicks.length);

    auto lat = Lattice!HAS_COIN.create();

    enum double SIGMA_RANGE = 4.0;   // extend chains to this many sigma
    enum int CHAIN_PAD = 5;          // extra sites beyond sigma range
    enum double GRID_PAD = 5.0;      // extra extent for density grid

    int maxChainLen = cast(int)(SIGMA_RANGE * p.sigma / STEP_LEN) + CHAIN_PAD;
    double gridHalf = maxChainLen * STEP_LEN + GRID_PAD;
    auto grid = ProximityGrid.create(gridHalf, p.dMin);

    stderr.writefln("\n--- Chain-first site generation (dMin=%.3f, grid %d^3) ---",
                    p.dMin, grid.gridN);
    int nChains = generateSites(lat, p.sigma, p.seedThresh, grid);

    int noR = 0, noL = 0;
    foreach (s; 0 .. lat.nsites) {
        if (!lat.hasChain(s, IS_R)) noR++;
        if (!lat.hasChain(s, IS_L)) noL++;
    }
    stderr.writefln("Seed: %d sites, %d chains", lat.nsites, nChains);
    stderr.writefln("Coverage: %d without R-chain, %d without L-chain", noR, noL);

    // A4 symmetry diagnostic
    checkA4Symmetry(lat, grid, 30);

    // Snapshot seed lattice so we can reuse it for multiple k0 values
    auto snap = lat.takeSnapshot();
    stderr.writefln("Snapshot saved (%d sites, %d chains)", snap.nsites, snap.chainSiteIds.length);

    foreach (ki, k0; p.kicks) {
    lat.restoreSnapshot(snap);

    initWavepacket(lat, p.sigma, k0.x, k0.y, k0.z);
    stderr.writefln("\n--- Run %d/%d: k0=(%.4f,%.4f,%.4f) ---",
                    ki + 1, p.kicks.length, k0.x, k0.y, k0.z);

    writefln("# theta=%.4f sigma=%.1f n_steps=%d ext_thresh=%.1e prune_thresh=%.1e mix_phi=%.4f k0=(%.4f,%.4f,%.4f)",
             p.theta, p.sigma, p.nSteps, p.threshold, p.pruneThresh, p.mixPhi,
             k0.x, k0.y, k0.z);
    writefln("# t norm xmean ymean zmean r2 x2 y2 z2 r95 nsites absorbed pruned");

    double totalAbsorbed = 0;
    double totalPruned = 0;

    foreach (t; 0 .. p.nSteps + 1) {
        import core.time : MonoTime;
        auto tObsStart = MonoTime.currTime;
        auto obs = computeObservables(lat);
        auto tObsEnd = MonoTime.currTime;

        writefln("%d %.6f %.6f %.6f %.6f %.2f %.2f %.2f %.2f %.2f %d %.6e %.6e",
                 t, obs.normPsi, obs.xMean, obs.yMean, obs.zMean,
                 obs.r2, obs.x2, obs.y2, obs.z2,
                 obs.r95, obs.nsites, totalAbsorbed, totalPruned);
        stdout.flush();

        if (t < p.nSteps) {
            // Check free RAM before each step's allocations
            if (isRamLow()) {
                stderr.writefln("  ABORT step %d: free RAM below 1 GB (%d MB free), stopping to prevent OOM",
                                t, freeRamBytes() / (1024 * 1024));
                break;
            }

            enum STATUS_INTERVAL = 5;
            if (t % STATUS_INTERVAL == 0)
                stderr.writefln("  step %d/%d: %d sites, norm=%.6f absorbed=%.2e pruned=%.2e",
                                t, p.nSteps, lat.nsites, obs.normPsi,
                                totalAbsorbed, totalPruned);

            auto t0 = MonoTime.currTime;

            // W = V_R · S_R · C_R · V_L · S_L · C_L
            // With HAS_COIN=false, applyCoin is a no-op (compiled out)
            applyCoin(lat, IS_L);
            auto tCoinL = MonoTime.currTime;
            auto resL = applyShift(lat, IS_L, extThresh2, pruneThresh2);
            auto tShiftL = MonoTime.currTime;
            applyVmix(lat, IS_L, p.mixPhi);
            auto tVmixL = MonoTime.currTime;
            applyCoin(lat, IS_R);
            auto tCoinR = MonoTime.currTime;
            auto resR = applyShift(lat, IS_R, extThresh2, pruneThresh2);
            auto tShiftR = MonoTime.currTime;
            applyVmix(lat, IS_R, p.mixPhi);
            auto tVmixR = MonoTime.currTime;

            totalAbsorbed += resL.probAbsorbed + resR.probAbsorbed;
            totalPruned += resL.probPruned + resR.probPruned;
            int nPruned = resL.nPruned + resR.nPruned;

            stderr.writefln("    S_L: absorbed=%.4e created=%d capFull=%d  S_R: absorbed=%.4e created=%d capFull=%d  sites=%d",
                            resL.probAbsorbed, resL.nCreated, resL.nCapFull,
                            resR.probAbsorbed, resR.nCreated, resR.nCapFull, lat.nsites);

            if (t % STATUS_INTERVAL == 0) {
                auto ms(MonoTime a, MonoTime b) { return (b - a).total!"msecs"; }
                stderr.writefln("    timing: obs=%dms coinL=%dms shiftL=%dms coinR=%dms shiftR=%dms pruned=%d",
                    ms(tObsStart, tObsEnd),
                    ms(t0, tCoinL), ms(tCoinL, tShiftL),
                    ms(tVmixL, tCoinR), ms(tCoinR, tShiftR), nPruned);
            }

            if (totalAbsorbed > 0.95) {
                stderr.writefln("  step %d: absorbed norm %.4f > 95%%, stopping",
                                t, totalAbsorbed);
                break;
            }

            int capFull = resL.nCapFull + resR.nCapFull;
            if (capFull > 0) {
                stderr.writefln("  ABORT step %d: lattice full (%d sites), %d extensions dropped — stopping",
                                t, lat.nsites, capFull);
                break;
            }
        }
    }

    stderr.writefln("\nDone k0=(%.4f,%.4f,%.4f): %d sites, absorbed=%.6e, pruned=%.6e",
                    k0.x, k0.y, k0.z, lat.nsites, totalAbsorbed, totalPruned);
    } // end k0 loop
}

version(walk_exe) {
    int main(string[] args) {
        auto p = parseArgs(args);
        run(p);
        return 0;
    }
}
