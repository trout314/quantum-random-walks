/**
 * walk_main.d — Main entry point for the 3D adaptive quantum walk.
 *
 * Parses command-line arguments, generates the lattice, initializes
 * the wavepacket, and runs the time evolution loop.
 */
module walk_main;

import std.math : sqrt, cos, sin, exp;
import std.complex : Complex;
import std.conv : to;
import std.stdio : writef, writefln, stderr, stdout;
import std.format : format;
import geometry : Vec3, dot;
import lattice : Lattice, DensityGrid, generateSites, PAT_R, PAT_L;
import operators : applyShift, applyCoin, applyVmix, pruneChainEnds, ShiftResult;
import observables : computeObservables, Observables;

alias C = Complex!double;

struct WalkParams {
    double theta = 0.5;
    double sigma = 3.0;
    int nSteps = 50;
    double threshold = 1e-10;
    int coinType = 1;      // 0 = e·α, 1 = dual parity
    double mixPhi = 0.0;
    int pruneInterval = 0;
}

WalkParams parseArgs(string[] args) {
    WalkParams p;
    if (args.length > 1) p.theta = args[1].to!double;
    if (args.length > 2) p.sigma = args[2].to!double;
    if (args.length > 3) p.nSteps = args[3].to!int;
    if (args.length > 4) p.threshold = args[4].to!double;
    if (args.length > 5) p.coinType = args[5].to!int;
    if (args.length > 6) p.mixPhi = args[6].to!double;
    if (args.length > 7) p.pruneInterval = args[7].to!int;
    return p;
}

void initWavepacket(ref Lattice lat, double sigma) {
    double norm2 = 0;
    foreach (n; 0 .. lat.nsites) {
        double r2 = dot(lat.sites[n].pos, lat.sites[n].pos);
        double w = exp(-r2 / (2 * sigma * sigma));
        lat.psi[4*n] = C(w, 0);
        norm2 += w * w;
    }
    double nf = 1.0 / sqrt(norm2);
    foreach (i; 0 .. 4 * lat.nsites)
        lat.psi[i] = C(nf, 0) * lat.psi[i];
}

void run(WalkParams p) {
    double thresh2 = p.threshold * p.threshold;
    double ct = cos(p.theta), st = sin(p.theta);

    stderr.writefln("=== walk_d: theta=%.3f sigma=%.1f steps=%d thresh=%.1e coin=%s mix_phi=%.4f prune=%d ===",
                    p.theta, p.sigma, p.nSteps, p.threshold,
                    p.coinType == 0 ? "e.alpha" : "dual_parity",
                    p.mixPhi, p.pruneInterval);

    // Generate lattice
    enum int MAX_SITES = 60_000_000;
    enum int HASH_BITS = 27;
    auto lat = Lattice.create(MAX_SITES, HASH_BITS);

    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * p.sigma / stepLen) + 5;
    double gridHalf = maxChainLen * stepLen + 5.0;
    auto grid = DensityGrid.create(gridHalf, 8);

    stderr.writefln("\n--- Chain-first site generation (grid %d³) ---", grid.gridN);
    int nChains = generateSites(lat, p.sigma, 1e-4, grid);

    // Chain coverage report
    int noR = 0, noL = 0;
    foreach (s; 0 .. lat.nsites) {
        if (lat.chainFace(s, true) < 0) noR++;
        if (lat.chainFace(s, false) < 0) noL++;
    }
    stderr.writefln("Seed: %d sites, %d chains", lat.nsites, nChains);
    stderr.writefln("Coverage: %d without R-chain, %d without L-chain", noR, noL);

    // Initialize wavepacket
    initWavepacket(lat, p.sigma);
    stderr.writefln("Wavepacket initialized (sigma=%.1f)", p.sigma);

    // Output header
    writefln("# theta=%.4f sigma=%.1f n_steps=%d thresh=%.1e prune=%d",
             p.theta, p.sigma, p.nSteps, p.threshold, p.pruneInterval);
    writefln("# t norm r2 x2 y2 z2 r95 nsites absorbed pruned");

    double totalAbsorbed = 0;
    double totalPruned = 0;

    // Time evolution
    foreach (t; 0 .. p.nSteps + 1) {
        auto obs = computeObservables(lat);

        writefln("%d %.6f %.2f %.2f %.2f %.2f %.2f %d %.6e %.6e",
                 t, obs.normPsi, obs.r2, obs.x2, obs.y2, obs.z2,
                 obs.r95, obs.nsites, totalAbsorbed, totalPruned);
        stdout.flush();

        if (t < p.nSteps) {
            if (t % 5 == 0)
                stderr.writefln("  step %d/%d: %d sites, norm=%.6f absorbed=%.2e pruned=%.2e",
                                t, p.nSteps, lat.nsites, obs.normPsi,
                                totalAbsorbed, totalPruned);

            // W = V_R · S_R · C_R · V_L · S_L · C_L
            applyCoin(lat, false, ct, st, p.coinType);
            auto resL = applyShift(lat, false, PAT_L, thresh2);
            applyVmix(lat, false, p.mixPhi);
            applyCoin(lat, true, ct, st, p.coinType);
            auto resR = applyShift(lat, true, PAT_R, thresh2);
            applyVmix(lat, true, p.mixPhi);

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

// ---- C interface for Python ----

export extern(C)
int walk_run(int argc, const(char)** argv) {
    string[] args;
    foreach (i; 0 .. argc)
        args ~= to!string(argv[i][0 .. _cstrlen(argv[i])]);
    auto p = parseArgs(args);
    run(p);
    return 0;
}

private size_t _cstrlen(const(char)* s) {
    size_t n = 0;
    while (s[n] != 0) n++;
    return n;
}

// ---- Standalone entry point ----
// This is only used when building as an executable (not as shared lib).

version(walk_exe) {
    int main(string[] args) {
        auto p = parseArgs(args);
        run(p);
        return 0;
    }
}
