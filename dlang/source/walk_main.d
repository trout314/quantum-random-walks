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
import lattice : Lattice, DensityGrid, generateSites, PAT_R, PAT_L, IS_R, IS_L;
import operators : applyShift, applyCoin, applyVmix, pruneChainEnds, ShiftResult;
import observables : computeObservables, Observables;

enum bool HAS_COIN = false;

struct WalkParams {
    double theta = 0.0;        // default: no coin (V-mixing only)
    double sigma = 3.0;
    int nSteps = 50;
    double threshold = 1e-10;  // extension threshold (absorb if amp < this)
    double pruneThresh = 1e-6; // pruning threshold (prune chain ends below this)
    double mixPhi = 0.0;
    int pruneInterval = 10;
    int maxSites = 25_000_000; // lattice capacity (~24 GB peak for sigma=5)
    double seedThresh = 1e-4;  // amplitude cutoff for site generation
    int maxDensity = 2;        // max sites per 1x1x1 grid cell
    double k0x = 0.0;         // momentum kick along x
    double k0y = 0.0;         // momentum kick along y
    double k0z = 0.0;         // momentum kick along z
}

WalkParams parseArgs(string[] args) {
    WalkParams p;
    if (args.length > 1) p.theta = args[1].to!double;
    if (args.length > 2) p.sigma = args[2].to!double;
    if (args.length > 3) p.nSteps = args[3].to!int;
    if (args.length > 4) p.threshold = args[4].to!double;
    if (args.length > 5) p.pruneThresh = args[5].to!double;
    if (args.length > 6) p.mixPhi = args[6].to!double;
    if (args.length > 7) p.pruneInterval = args[7].to!int;
    if (args.length > 8) p.maxSites = args[8].to!int;
    if (args.length > 9) p.seedThresh = args[9].to!double;
    if (args.length > 10) p.maxDensity = args[10].to!int;
    if (args.length > 11) p.k0x = args[11].to!double;
    if (args.length > 12) p.k0y = args[12].to!double;
    if (args.length > 13) p.k0z = args[13].to!double;
    return p;
}

void initWavepacket(ref Lattice!HAS_COIN lat, double sigma,
                    double k0x, double k0y, double k0z) {
    double norm2 = 0;
    foreach (n; 0 .. lat.nsites) {
        double r2 = dot(lat.sites[n].pos, lat.sites[n].pos);
        double w = exp(-r2 / (2 * sigma * sigma));
        double phase = k0x * lat.sites[n].pos.x
                     + k0y * lat.sites[n].pos.y
                     + k0z * lat.sites[n].pos.z;
        double phRe = cos(phase);
        double phIm = sin(phase);
        lat.psiRe[4*n] = w * phRe;
        lat.psiIm[4*n] = w * phIm;
        norm2 += w * w;  // |w * e^{i phase}|^2 = w^2
    }
    double nf = 1.0 / sqrt(norm2);
    foreach (i; 0 .. 4 * lat.nsites) {
        lat.psiRe[i] *= nf;
        lat.psiIm[i] *= nf;
    }
}

void run(WalkParams p) {
    double extThresh2 = p.threshold * p.threshold;
    double pruneThresh2 = p.pruneThresh * p.pruneThresh;

    stderr.writefln("=== walk_d: theta=%.3f sigma=%.1f steps=%d ext_thresh=%.1e prune_thresh=%.1e mix_phi=%.4f prune=%d k0=(%.4f,%.4f,%.4f) ===",
                    p.theta, p.sigma, p.nSteps, p.threshold, p.pruneThresh, p.mixPhi, p.pruneInterval,
                    p.k0x, p.k0y, p.k0z);

    auto lat = Lattice!HAS_COIN.create(p.maxSites);

    enum double SIGMA_RANGE = 4.0;   // extend chains to this many sigma
    enum int CHAIN_PAD = 5;          // extra sites beyond sigma range
    enum double GRID_PAD = 5.0;      // extra extent for density grid

    int maxChainLen = cast(int)(SIGMA_RANGE * p.sigma / STEP_LEN) + CHAIN_PAD;
    double gridHalf = maxChainLen * STEP_LEN + GRID_PAD;
    auto grid = DensityGrid.create(gridHalf, p.maxDensity);

    stderr.writefln("\n--- Chain-first site generation (grid %d^3) ---", grid.gridN);
    int nChains = generateSites(lat, p.sigma, p.seedThresh, grid);

    int noR = 0, noL = 0;
    foreach (s; 0 .. lat.nsites) {
        if (lat.chainFace(s, IS_R) < 0) noR++;
        if (lat.chainFace(s, IS_L) < 0) noL++;
    }
    stderr.writefln("Seed: %d sites, %d chains", lat.nsites, nChains);
    stderr.writefln("Coverage: %d without R-chain, %d without L-chain", noR, noL);

    initWavepacket(lat, p.sigma, p.k0x, p.k0y, p.k0z);
    stderr.writefln("Wavepacket initialized (sigma=%.1f, k0=(%.4f,%.4f,%.4f))",
                    p.sigma, p.k0x, p.k0y, p.k0z);

    writefln("# theta=%.4f sigma=%.1f n_steps=%d ext_thresh=%.1e prune_thresh=%.1e mix_phi=%.4f prune=%d k0=(%.4f,%.4f,%.4f)",
             p.theta, p.sigma, p.nSteps, p.threshold, p.pruneThresh, p.mixPhi, p.pruneInterval,
             p.k0x, p.k0y, p.k0z);
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
            auto resL = applyShift(lat, IS_L, PAT_L, extThresh2);
            auto tShiftL = MonoTime.currTime;
            applyVmix(lat, IS_L, p.mixPhi);
            auto tVmixL = MonoTime.currTime;
            applyCoin(lat, IS_R);
            auto tCoinR = MonoTime.currTime;
            auto resR = applyShift(lat, IS_R, PAT_R, extThresh2);
            auto tShiftR = MonoTime.currTime;
            applyVmix(lat, IS_R, p.mixPhi);
            auto tVmixR = MonoTime.currTime;

            if (t % STATUS_INTERVAL == 0) {
                auto ms(MonoTime a, MonoTime b) { return (b - a).total!"msecs"; }
                stderr.writefln("    timing: obs=%dms coinL=%dms shiftL=%dms coinR=%dms shiftR=%dms",
                    ms(tObsStart, tObsEnd),
                    ms(t0, tCoinL), ms(tCoinL, tShiftL),
                    ms(tVmixL, tCoinR), ms(tCoinR, tShiftR));
            }

            totalAbsorbed += resL.probAbsorbed + resR.probAbsorbed;

            if (totalAbsorbed > 0.05) {
                stderr.writefln("  step %d: absorbed norm %.4f > 5%%, stopping",
                                t, totalAbsorbed);
                break;
            }

            int capFull = resL.nCapFull + resR.nCapFull;
            if (capFull > 0)
                stderr.writefln("  WARNING step %d: lattice full (%d sites), %d extensions dropped",
                                t, lat.nsites, capFull);

            if (p.pruneInterval > 0 && t > 0 && t % p.pruneInterval == 0) {
                auto pr = pruneChainEnds(lat, pruneThresh2);
                totalPruned += pr.probPruned;
                if (pr.count > 0)
                    stderr.writefln("  step %d: pruned %d sites (free=%d, prob=%.2e)",
                                    t, pr.count, lat.freeCount, pr.probPruned);
            }
        }
    }

    stderr.writefln("\nDone. Final: %d sites, absorbed=%.6e, pruned=%.6e",
                    lat.nsites, totalAbsorbed, totalPruned);
}

version(walk_exe) {
    int main(string[] args) {
        auto p = parseArgs(args);
        run(p);
        return 0;
    }
}
