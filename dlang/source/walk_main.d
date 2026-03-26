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
import lattice : Lattice, ProximityGrid, generateSites, PAT_R, PAT_L, IS_R, IS_L, isRamLow, freeRamBytes;
import dirac : Mat4, makeTau, frameTransport, matVecSplit, projPlus, projMinus;
import operators : applyShift, applyCoin, applyVmix, ShiftResult;
import observables : computeObservables, Observables;

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
    int face = lat.chainFace(s, isR);
    return makeTau(lat.sites[s].dirs[face]);
}

void initWavepacket(ref Lattice!HAS_COIN lat, double sigma,
                    double k0x, double k0y, double k0z) {
    int ns = lat.nsites;

    // BFS to compute frame-transported reference spinor at each site.
    // refRe/refIm[4*n..4*n+4] holds U(origin→n) · (1,0,0,0).
    auto refRe = new double[4 * ns];  refRe[] = 0;
    auto refIm = new double[4 * ns];  refIm[] = 0;
    auto visited = new bool[ns];      visited[] = false;

    // Build P+/P- symmetric reference spinor at the origin.
    // Project (1,0,0,0) onto P+ and P-, normalize each, sum with equal weight.
    {
        bool originIsR = lat.chainFace(0, true) >= 0;
        Mat4 tau0 = siteTau(lat, 0, originIsR);
        Mat4 Pp = projPlus(tau0);
        Mat4 Pm = projMinus(tau0);

        double[4] baseRe = [1, 0, 0, 0];
        double[4] baseIm = [0, 0, 0, 0];

        // P+ component
        double[4] ppRe = 0, ppIm = 0;
        matVecSplit(Pp, baseRe.ptr, baseIm.ptr, ppRe.ptr, ppIm.ptr);
        double ppNorm2 = 0;
        foreach (a; 0 .. 4) ppNorm2 += ppRe[a]*ppRe[a] + ppIm[a]*ppIm[a];

        // P- component
        double[4] pmRe = 0, pmIm = 0;
        matVecSplit(Pm, baseRe.ptr, baseIm.ptr, pmRe.ptr, pmIm.ptr);
        double pmNorm2 = 0;
        foreach (a; 0 .. 4) pmNorm2 += pmRe[a]*pmRe[a] + pmIm[a]*pmIm[a];

        // Normalize each and sum: ref = (|p+⟩/||p+|| + |p-⟩/||p-||) / √2
        double ppInv = 1.0 / sqrt(ppNorm2);
        double pmInv = 1.0 / sqrt(pmNorm2);
        double scale = 1.0 / sqrt(2.0);
        foreach (a; 0 .. 4) {
            refRe[a] = scale * (ppInv * ppRe[a] + pmInv * pmRe[a]);
            refIm[a] = scale * (ppInv * ppIm[a] + pmInv * pmIm[a]);
        }

        stderr.writefln("  IC: P+/P- symmetric (P+ weight=%.1f%%, P- weight=%.1f%%)",
                        50.0, 50.0);
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
            if (lat.chainFace(s, isR) < 0) continue;
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
        norm2 += w * w;  // |w * e^{ik} * ref|^2 = w^2 * |ref|^2 = w^2 (ref is unit)
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
        if (lat.chainFace(s, IS_R) < 0) noR++;
        if (lat.chainFace(s, IS_L) < 0) noL++;
    }
    stderr.writefln("Seed: %d sites, %d chains", lat.nsites, nChains);
    stderr.writefln("Coverage: %d without R-chain, %d without L-chain", noR, noL);

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
            auto resL = applyShift(lat, IS_L, PAT_L, extThresh2, pruneThresh2);
            auto tShiftL = MonoTime.currTime;
            applyVmix(lat, IS_L, p.mixPhi);
            auto tVmixL = MonoTime.currTime;
            applyCoin(lat, IS_R);
            auto tCoinR = MonoTime.currTime;
            auto resR = applyShift(lat, IS_R, PAT_R, extThresh2, pruneThresh2);
            auto tShiftR = MonoTime.currTime;
            applyVmix(lat, IS_R, p.mixPhi);
            auto tVmixR = MonoTime.currTime;

            totalAbsorbed += resL.probAbsorbed + resR.probAbsorbed;
            totalPruned += resL.probPruned + resR.probPruned;
            int nPruned = resL.nPruned + resR.nPruned;

            if (t % STATUS_INTERVAL == 0) {
                auto ms(MonoTime a, MonoTime b) { return (b - a).total!"msecs"; }
                stderr.writefln("    timing: obs=%dms coinL=%dms shiftL=%dms coinR=%dms shiftR=%dms pruned=%d",
                    ms(tObsStart, tObsEnd),
                    ms(t0, tCoinL), ms(tCoinL, tShiftL),
                    ms(tVmixL, tCoinR), ms(tCoinR, tShiftR), nPruned);
            }

            if (totalAbsorbed > 0.05) {
                stderr.writefln("  step %d: absorbed norm %.4f > 5%%, stopping",
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
