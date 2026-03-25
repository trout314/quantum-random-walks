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
    double threshold = 1e-10;
    double mixPhi = 0.0;
    int pruneInterval = 0;
    int maxSites = 60_000_000; // lattice capacity
    double seedThresh = 1e-4;  // amplitude cutoff for site generation
}

WalkParams parseArgs(string[] args) {
    WalkParams p;
    if (args.length > 1) p.theta = args[1].to!double;
    if (args.length > 2) p.sigma = args[2].to!double;
    if (args.length > 3) p.nSteps = args[3].to!int;
    if (args.length > 4) p.threshold = args[4].to!double;
    if (args.length > 5) p.mixPhi = args[5].to!double;
    if (args.length > 6) p.pruneInterval = args[6].to!int;
    if (args.length > 7) p.maxSites = args[7].to!int;
    if (args.length > 8) p.seedThresh = args[8].to!double;
    return p;
}

void initWavepacket(ref Lattice!HAS_COIN lat, double sigma) {
    double norm2 = 0;
    foreach (n; 0 .. lat.nsites) {
        double r2 = dot(lat.sites[n].pos, lat.sites[n].pos);
        double w = exp(-r2 / (2 * sigma * sigma));
        lat.psiRe[4*n] = w;
        norm2 += w * w;
    }
    double nf = 1.0 / sqrt(norm2);
    foreach (i; 0 .. 4 * lat.nsites)
        lat.psiRe[i] *= nf;
}

void run(WalkParams p) {
    double thresh2 = p.threshold * p.threshold;

    stderr.writefln("=== walk_d: theta=%.3f sigma=%.1f steps=%d thresh=%.1e mix_phi=%.4f prune=%d ===",
                    p.theta, p.sigma, p.nSteps, p.threshold, p.mixPhi, p.pruneInterval);

    auto lat = Lattice!HAS_COIN.create(p.maxSites);

    enum double SIGMA_RANGE = 4.0;   // extend chains to this many sigma
    enum int CHAIN_PAD = 5;          // extra sites beyond sigma range
    enum double GRID_PAD = 5.0;      // extra extent for density grid
    enum int MAX_DENSITY_PER_CELL = 8;

    int maxChainLen = cast(int)(SIGMA_RANGE * p.sigma / STEP_LEN) + CHAIN_PAD;
    double gridHalf = maxChainLen * STEP_LEN + GRID_PAD;
    auto grid = DensityGrid.create(gridHalf, MAX_DENSITY_PER_CELL);

    stderr.writefln("\n--- Chain-first site generation (grid %d^3) ---", grid.gridN);
    int nChains = generateSites(lat, p.sigma, p.seedThresh, grid);

    int noR = 0, noL = 0;
    foreach (s; 0 .. lat.nsites) {
        if (lat.chainFace(s, IS_R) < 0) noR++;
        if (lat.chainFace(s, IS_L) < 0) noL++;
    }
    stderr.writefln("Seed: %d sites, %d chains", lat.nsites, nChains);
    stderr.writefln("Coverage: %d without R-chain, %d without L-chain", noR, noL);

    initWavepacket(lat, p.sigma);
    stderr.writefln("Wavepacket initialized (sigma=%.1f)", p.sigma);

    writefln("# theta=%.4f sigma=%.1f n_steps=%d thresh=%.1e mix_phi=%.4f prune=%d",
             p.theta, p.sigma, p.nSteps, p.threshold, p.mixPhi, p.pruneInterval);
    writefln("# t norm r2 x2 y2 z2 r95 nsites absorbed pruned");

    double totalAbsorbed = 0;
    double totalPruned = 0;

    foreach (t; 0 .. p.nSteps + 1) {
        import core.time : MonoTime;
        auto tObsStart = MonoTime.currTime;
        auto obs = computeObservables(lat);
        auto tObsEnd = MonoTime.currTime;

        writefln("%d %.6f %.2f %.2f %.2f %.2f %.2f %d %.6e %.6e",
                 t, obs.normPsi, obs.r2, obs.x2, obs.y2, obs.z2,
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
            auto resL = applyShift(lat, IS_L, PAT_L, thresh2);
            auto tShiftL = MonoTime.currTime;
            applyVmix(lat, IS_L, p.mixPhi);
            auto tVmixL = MonoTime.currTime;
            applyCoin(lat, IS_R);
            auto tCoinR = MonoTime.currTime;
            auto resR = applyShift(lat, IS_R, PAT_R, thresh2);
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

            if (p.pruneInterval > 0 && t > 0 && t % p.pruneInterval == 0) {
                auto pr = pruneChainEnds(lat, thresh2);
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
