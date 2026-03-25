/**
 * walk_1d.d — 1D quantum walk on a single BC helix chain.
 *
 * Adaptive: chain extends on-the-fly as the wavepacket spreads.
 * Pre-computes shift blocks, coin operators, and V-mixing at each site.
 */
module walk_1d;

import std.math : sqrt, cos, sin, exp, fabs;
import std.complex : Complex;
import std.conv : to;
import std.stdio : writef, writefln, stderr, stdout, File;
import geometry : Vec3, dot, norm, helixStep, reorth, initTet;
import dirac : Mat4, C, makeTau, projPlus, projMinus, frameTransport, mul, conj, alpha;

enum MAX_N = 500_000;

struct Walk1dParams {
    double theta = 0.5;
    double sigma = 50.0;
    int nSteps = 500;
    int coinType = 3;      // 3 = dual parity (default)
    int nuType = 0;
    double k0 = 0.0;       // initial momentum kick
    double mixPhi = 0.0;   // post-shift mixing angle
    int spiralType = 0;    // 0=R, 1=L
}

Walk1dParams parseArgs1d(string[] args) {
    Walk1dParams p;
    if (args.length > 1) p.theta = args[1].to!double;
    if (args.length > 2) p.sigma = args[2].to!double;
    if (args.length > 3) p.nSteps = args[3].to!int;
    if (args.length > 4) p.coinType = args[4].to!int;
    if (args.length > 5) p.nuType = args[5].to!int;
    if (args.length > 6) p.k0 = args[6].to!double;
    if (args.length > 7) p.mixPhi = args[7].to!double;
    if (args.length > 8) p.spiralType = args[8].to!int;
    return p;
}

/// All per-site precomputed data for the 1D chain.
struct Chain1d {
    Vec3[MAX_N] pos;
    Vec3[4][MAX_N] dirs;
    int[MAX_N] faceIdx;
    Mat4[MAX_N] tau;
    Mat4[MAX_N] Pp, Pm;
    Mat4[MAX_N] fwdBlock, bwdBlock;
    Mat4[MAX_N] coin1, coin2, coin3;
    Mat4[MAX_N] vmix;
    int builtUpTo;

    int[4] pat;
    double ct, st;
    int coinType;
    bool useCoin2, useCoin3;
    double mixPhi;
}

// These are too large for the stack; allocate on the heap.
Chain1d* newChain1d() {
    import core.stdc.stdlib : calloc;
    auto ch = cast(Chain1d*) calloc(1, Chain1d.sizeof);
    ch.builtUpTo = 0;
    ch.useCoin2 = false;
    ch.useCoin3 = false;
    ch.mixPhi = 0.0;
    return ch;
}

/// Build chain geometry and operators up to and including site i.
void ensureSite(Chain1d* ch, int i) {
    assert(i >= 0 && i < MAX_N, "site out of range");
    while (ch.builtUpTo <= i) {
        int n = ch.builtUpTo;
        if (n == 0) {
            ch.dirs[0] = initTet();
            ch.pos[0] = Vec3(0, 0, 0);
            ch.faceIdx[0] = ch.pat[0];
        } else {
            ch.dirs[n] = ch.dirs[n-1];
            ch.pos[n] = ch.pos[n-1];
            helixStep(ch.pos[n], ch.dirs[n], ch.pat[(n-1) % 4]);
            if (n % 8 == 0) reorth(ch.dirs[n]);
            ch.faceIdx[n] = ch.pat[n % 4];
        }

        // tau and projectors
        ch.tau[n] = makeTau(ch.dirs[n][ch.faceIdx[n]]);
        ch.Pp[n] = projPlus(ch.tau[n]);
        ch.Pm[n] = projMinus(ch.tau[n]);

        // Shift blocks
        if (n > 0) {
            Mat4 U = frameTransport(ch.tau[n], ch.tau[n-1]);
            ch.bwdBlock[n] = mul(U, ch.Pm[n]);
            U = frameTransport(ch.tau[n-1], ch.tau[n]);
            ch.fwdBlock[n-1] = mul(U, ch.Pp[n-1]);
        }

        // Coins
        Vec3 e = ch.dirs[n][ch.faceIdx[n]];
        double en = norm(e);
        Vec3 ehat = e * (1.0 / en);

        if (ch.coinType == 3 || ch.coinType == 4 || ch.coinType == 5) {
            // Dual parity: f1, f2 perpendicular to e
            Vec3 f1 = Vec3(ehat.y, -ehat.x, 0);
            double fn = norm(f1);
            if (fn < 1e-10) { f1 = Vec3(-ehat.z, 0, ehat.x); fn = norm(f1); }
            f1 = f1 * (1.0 / fn);
            Vec3 f2 = Vec3(ehat.y*f1.z - ehat.z*f1.y,
                           ehat.z*f1.x - ehat.x*f1.z,
                           ehat.x*f1.y - ehat.y*f1.x);
            fn = norm(f2); f2 = f2 * (1.0 / fn);

            ch.coin1[n] = buildCoinMatrix([f1.x, f1.y, f1.z], ch.ct, ch.st);
            ch.coin2[n] = buildCoinMatrix([f2.x, f2.y, f2.z], ch.ct, ch.st);
            ch.useCoin2 = true;

            if (ch.coinType == 4) {
                // Beta coin
                ch.coin3[n] = buildBetaCoin(ch.ct, ch.st);
                ch.useCoin3 = true;
            } else if (ch.coinType == 5) {
                // e·α coin
                ch.coin3[n] = buildCoinMatrix([e.x, e.y, e.z], ch.ct, ch.st);
                ch.useCoin3 = true;
            }
        } else if (ch.coinType == 1) {
            ch.coin1[n] = buildCoinMatrix([e.x, e.y, e.z], ch.ct, ch.st);
        } else {
            ch.coin1[n] = buildBetaCoin(ch.ct, ch.st);
        }

        // V-mixing
        if (ch.mixPhi != 0.0) {
            ch.vmix[n] = buildVmix(ch.Pp[n], ch.Pm[n], ch.mixPhi);
        } else {
            ch.vmix[n] = Mat4.eye();
        }

        ch.builtUpTo++;
    }
}

/// Build coin matrix: exp(-iθ (d·α)) = cos(θ)I - i sin(θ)(d·α)
Mat4 buildCoinMatrix(double[3] d, double ct, double st) {
    auto m = Mat4(0);
    foreach (a; 0 .. 4)
        foreach (b; 0 .. 4) {
            C ea = C(0, 0);
            foreach (k; 0 .. 3)
                ea = ea + C(d[k], 0) * alpha(k)[a, b];
            m[a, b] = C(ct * (a == b ? 1.0 : 0.0), 0) - C(0, st) * ea;
        }
    return m;
}

/// Build beta coin: exp(-iθ β)
Mat4 buildBetaCoin(double ct, double st) {
    auto m = Mat4(0);
    double[4] bd = [1, 1, -1, -1];
    foreach (a; 0 .. 4)
        m[a, a] = C(ct, -st * bd[a]);
    return m;
}

/// Build V-mixing operator from projectors.
Mat4 buildVmix(Mat4 Pp, Mat4 Pm, double mixPhi) {
    double cp = cos(mixPhi), sp = sin(mixPhi);

    // Gram-Schmidt for P+ and P- bases
    C[4][2] ppBasis, pmBasis;
    int npFound = 0, nmFound = 0;
    foreach (col; 0 .. 4) {
        if (npFound >= 2 && nmFound >= 2) break;
        if (npFound < 2) {
            C[4] v;
            foreach (a; 0 .. 4) v[a] = Pp[a, col];
            foreach (j; 0 .. npFound) {
                C d = C(0,0);
                foreach (a; 0 .. 4) d = d + conj(ppBasis[a][j]) * v[a];
                foreach (a; 0 .. 4) v[a] = v[a] - d * ppBasis[a][j];
            }
            double nm = 0;
            foreach (a; 0 .. 4) nm += v[a].re * v[a].re + v[a].im * v[a].im;
            if (nm > 1e-10) {
                double inv = 1.0 / sqrt(nm);
                foreach (a; 0 .. 4) ppBasis[a][npFound] = C(inv, 0) * v[a];
                npFound++;
            }
        }
        if (nmFound < 2) {
            C[4] v;
            foreach (a; 0 .. 4) v[a] = Pm[a, col];
            foreach (j; 0 .. nmFound) {
                C d = C(0,0);
                foreach (a; 0 .. 4) d = d + conj(pmBasis[a][j]) * v[a];
                foreach (a; 0 .. 4) v[a] = v[a] - d * pmBasis[a][j];
            }
            double nm = 0;
            foreach (a; 0 .. 4) nm += v[a].re * v[a].re + v[a].im * v[a].im;
            if (nm > 1e-10) {
                double inv = 1.0 / sqrt(nm);
                foreach (a; 0 .. 4) pmBasis[a][nmFound] = C(inv, 0) * v[a];
                nmFound++;
            }
        }
    }

    auto V = Mat4(0);
    foreach (j; 0 .. 2)
        foreach (a; 0 .. 4)
            foreach (b; 0 .. 4) {
                V[a, b] = V[a, b] + pmBasis[a][j] * conj(ppBasis[b][j]);
                V[a, b] = V[a, b] + ppBasis[a][j] * conj(pmBasis[b][j]);
            }
    foreach (a; 0 .. 4)
        foreach (b; 0 .. 4)
            V[a, b] = C(cp * (a == b ? 1.0 : 0.0), 0) + C(0, sp) * V[a, b];
    return V;
}

/// Apply a precomputed 4×4 matrix to spinor at site i, in place.
void applyMat4InPlace(Mat4 M, C[] psi, int i) {
    C[4] result = [C(0,0), C(0,0), C(0,0), C(0,0)];
    foreach (a; 0 .. 4)
        foreach (b; 0 .. 4)
            result[a] = result[a] + M[a, b] * psi[4*i + b];
    psi[4*i .. 4*i+4] = result[];
}

void run1d(Walk1dParams p) {
    double ct = cos(p.theta), st = sin(p.theta);

    int[4] pat = p.spiralType == 1
        ? [0, 1, 2, 3]   // L helix
        : [1, 3, 0, 2];  // R helix

    stderr.writefln("=== walk_1d_d: theta=%.3f sigma=%.1f steps=%d coin=%d k0=%.4f spiral=%s ===",
                    p.theta, p.sigma, p.nSteps, p.coinType, p.k0,
                    p.spiralType == 1 ? "L" : "R");

    auto ch = newChain1d();
    ch.pat = pat;
    ch.ct = ct;
    ch.st = st;
    ch.coinType = p.coinType;
    ch.mixPhi = p.mixPhi;

    // Initialize wavepacket centered at MAX_N/2
    int center = MAX_N / 2;
    int lo = center - cast(int)(4 * p.sigma) - 10;
    int hi = center + cast(int)(4 * p.sigma) + 10;
    if (lo < 0) lo = 0;
    if (hi >= MAX_N) hi = MAX_N - 1;
    ensureSite(ch, hi);

    auto psi = new C[4 * MAX_N];
    psi[] = C(0, 0);

    // Frame-transported Gaussian IC with momentum kick
    double norm2 = 0;

    // Forward from center
    {
        C[4] chiCur = [C(1,0), C(0,0), C(0,0), C(0,0)];
        foreach (i; center .. hi + 1) {
            double x = cast(double)(i - center);
            double w = exp(-x*x / (2 * p.sigma * p.sigma));
            C phase = C(cos(p.k0 * x), sin(p.k0 * x));
            foreach (a; 0 .. 4)
                psi[4*i + a] = C(w, 0) * phase * chiCur[a];
            norm2 += w * w;
            if (i < hi) {
                ensureSite(ch, i + 1);
                Mat4 U = frameTransport(ch.tau[i], ch.tau[i+1]);
                C[4] chiNext = [C(0,0), C(0,0), C(0,0), C(0,0)];
                foreach (a; 0 .. 4)
                    foreach (b; 0 .. 4)
                        chiNext[a] = chiNext[a] + U[a, b] * chiCur[b];
                chiCur = chiNext;
            }
        }
    }

    // Backward from center
    {
        C[4] chiCur = [C(1,0), C(0,0), C(0,0), C(0,0)];
        foreach_reverse (i; lo .. center) {
            ensureSite(ch, i);
            Mat4 U = frameTransport(ch.tau[i+1], ch.tau[i]);
            C[4] chiNext = [C(0,0), C(0,0), C(0,0), C(0,0)];
            foreach (a; 0 .. 4)
                foreach (b; 0 .. 4)
                    chiNext[a] = chiNext[a] + U[a, b] * chiCur[b];
            chiCur = chiNext;
            double x = cast(double)(i - center);
            double w = exp(-x*x / (2 * p.sigma * p.sigma));
            C phase = C(cos(p.k0 * x), sin(p.k0 * x));
            foreach (a; 0 .. 4)
                psi[4*i + a] = C(w, 0) * phase * chiCur[a];
            norm2 += w * w;
        }
    }

    double nf = 1.0 / sqrt(norm2);
    foreach (i; lo .. hi + 1)
        foreach (a; 0 .. 4)
            psi[4*i + a] = C(nf, 0) * psi[4*i + a];

    int activeLo = lo, activeHi = hi + 1;
    immutable double ampThresh = 1e-15;

    stderr.writefln("Initial: active [%d, %d), center=%d, width=%d",
                    activeLo, activeHi, center, activeHi - activeLo);

    // Output header
    writefln("# theta=%.4f sigma=%.1f n_steps=%d coin=%d k0=%.6f",
             p.theta, p.sigma, p.nSteps, p.coinType, p.k0);
    writefln("# t norm x_mean x2 active_width");

    auto tmpPsi = new C[4 * MAX_N];

    foreach (t; 0 .. p.nSteps + 1) {
        // Observables
        double totalProb = 0, mx = 0;
        foreach (i; activeLo .. activeHi) {
            double prob = 0;
            foreach (a; 0 .. 4) {
                auto z = psi[4*i + a];
                prob += z.re * z.re + z.im * z.im;
            }
            totalProb += prob;
            mx += prob * cast(double)(i - center);
        }
        double pnorm = sqrt(totalProb);
        double xMean = mx / totalProb;

        // Variance using centered coordinates
        double var = 0;
        foreach (i; activeLo .. activeHi) {
            double prob = 0;
            foreach (a; 0 .. 4) {
                auto z = psi[4*i + a];
                prob += z.re * z.re + z.im * z.im;
            }
            double dx = cast(double)(i - center) - xMean;
            var += prob * dx * dx;
        }
        var /= totalProb;
        double mx2 = var + xMean * xMean;

        if (t % 10 == 0 || t <= 5)
            writefln("%d %.8f %.6f %.6f %d", t, pnorm, xMean, mx2, activeHi - activeLo);

        if (t < p.nSteps) {
            int alo = activeLo, ahi = activeHi;

            // Coin
            foreach (i; alo .. ahi)
                applyMat4InPlace(ch.coin1[i], psi, i);
            if (ch.useCoin2)
                foreach (i; alo .. ahi)
                    applyMat4InPlace(ch.coin2[i], psi, i);
            if (ch.useCoin3)
                foreach (i; alo .. ahi)
                    applyMat4InPlace(ch.coin3[i], psi, i);

            // Shift
            if (activeLo > 0) ensureSite(ch, activeLo - 1);
            ensureSite(ch, activeHi);

            int zlo = (activeLo > 0) ? activeLo - 1 : 0;
            int zhi = activeHi + 1;
            tmpPsi[4*zlo .. 4*zhi] = C(0, 0);

            foreach (i; zlo .. zhi) {
                foreach (a; 0 .. 4) {
                    C s = C(0, 0);
                    if (i-1 >= alo && i-1 < ahi)
                        foreach (b; 0 .. 4)
                            s = s + ch.fwdBlock[i-1][a, b] * psi[4*(i-1) + b];
                    if (i+1 >= alo && i+1 < ahi)
                        foreach (b; 0 .. 4)
                            s = s + ch.bwdBlock[i+1][a, b] * psi[4*(i+1) + b];
                    tmpPsi[4*i + a] = s;
                }
            }

            // V-mixing
            if (p.mixPhi != 0.0)
                foreach (i; zlo .. zhi)
                    applyMat4InPlace(ch.vmix[i], tmpPsi, i);

            // Update active region
            if (activeLo > 0) {
                double prob = 0;
                foreach (a; 0 .. 4) {
                    auto z = tmpPsi[4*(activeLo-1) + a];
                    prob += z.re * z.re + z.im * z.im;
                }
                if (prob > ampThresh) activeLo--;
            }
            {
                double prob = 0;
                foreach (a; 0 .. 4) {
                    auto z = tmpPsi[4*activeHi + a];
                    prob += z.re * z.re + z.im * z.im;
                }
                if (prob > ampThresh) activeHi++;
            }

            // Copy tmp to psi
            psi[4*activeLo .. 4*activeHi] = tmpPsi[4*activeLo .. 4*activeHi];
        }
    }

    // Output density
    {
        auto pf = File("/tmp/walk_1d_density.dat", "w");
        double total = 0;
        foreach (i; activeLo .. activeHi)
            foreach (a; 0 .. 4) {
                auto z = psi[4*i + a];
                total += z.re * z.re + z.im * z.im;
            }
        pf.writefln("# site position prob prob_plus prob_minus");
        foreach (i; activeLo .. activeHi) {
            double prob = 0;
            foreach (a; 0 .. 4) {
                auto z = psi[4*i + a];
                prob += z.re * z.re + z.im * z.im;
            }
            // P+ and P- components
            double pp = 0, pm = 0;
            foreach (a; 0 .. 4) {
                C sp = C(0,0), sm = C(0,0);
                foreach (b; 0 .. 4) {
                    C tab = ch.tau[i][a, b];
                    C dab = C(a == b ? 1.0 : 0.0, 0);
                    sp = sp + (dab + tab) * psi[4*i + b];
                    sm = sm + (dab - tab) * psi[4*i + b];
                }
                pp += sp.re * sp.re + sp.im * sp.im;
                pm += sm.re * sm.re + sm.im * sm.im;
            }
            pp *= 0.25; pm *= 0.25;
            double r = norm(ch.pos[i]);
            if (i < center) r = -r;
            pf.writefln("%d %.6f %.10e %.10e %.10e",
                        i - center, r, prob / total, pp / total, pm / total);
        }
        stderr.writefln("Density -> /tmp/walk_1d_density.dat (%d sites)", activeHi - activeLo);
    }

    import core.stdc.stdlib : free;
    free(ch);
}

version(walk_1d_exe) {
    int main(string[] args) {
        auto p = parseArgs1d(args);
        run1d(p);
        return 0;
    }
}
