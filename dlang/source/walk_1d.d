/**
 * walk_1d.d — 1D quantum walk on a single BC helix chain.
 *
 * Adaptive: chain extends on-the-fly as the wavepacket spreads.
 * Pre-computes shift blocks, coin operators, and V-mixing at each site.
 * Uses split real/imaginary storage (no Complex!double).
 */
module walk_1d;

import std.math : sqrt, cos, sin, exp, fabs;
import std.conv : to;
import std.stdio : writef, writefln, stderr, stdout, File;
import geometry : Vec3, dot, norm, helixStep, reorth, initTet, helixCentroid, helixVertexDirs, helixExitDir;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, matVecSplit, alpha;

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
    int icType = 0;        // 0=(1,0,0,0), 1=(1,0,1,0)/√2, 2=P+/P- symmetric
    double[4] gaugePhases = 0;  // site-dependent phase: gaugePhases[n % 4]
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
    if (args.length > 9) p.icType = args[9].to!int;
    if (args.length > 10) p.gaugePhases[0] = args[10].to!double;
    if (args.length > 11) p.gaugePhases[1] = args[11].to!double;
    if (args.length > 12) p.gaugePhases[2] = args[12].to!double;
    if (args.length > 13) p.gaugePhases[3] = args[13].to!double;
    return p;
}

/// Bundled per-site operators for cache locality (mirrors 3D SiteOps).
struct Site1dOps {
    Mat4 fwdBlock, bwdBlock;
    Mat4 coin1, coin2, coin3;
    Mat4 vmix;
}

/// All per-site precomputed data for the 1D chain.
/// Arrays are heap-allocated slices carved from a single GC block
/// to avoid a large .init blob in the binary.
struct Chain1d {
    Vec3[] pos;
    Vec3[4][] dirs;
    int[] faceIdx;
    Site1dOps[] ops;
    int builtUpTo;

    int[4] pat;
    double ct, st;
    int coinType;
    bool useCoin2, useCoin3;
    double mixPhi;
    double[4] gaugePhases;
}

/// Allocate all Chain1d arrays from a single GC buffer.
Chain1d* newChain1d() {
    enum size_t VEC3_SZ   = Vec3.sizeof;               // 24
    enum size_t DIR4_SZ   = (Vec3[4]).sizeof;           // 96
    enum size_t INT_SZ    = int.sizeof;                 // 4
    enum size_t OPS_SZ    = Site1dOps.sizeof;
    enum size_t TOTAL_PER = VEC3_SZ + DIR4_SZ + INT_SZ + OPS_SZ;

    auto buf = new ubyte[TOTAL_PER * MAX_N];
    buf[] = 0;

    // Carve slices from the buffer.
    size_t off = 0;

    T[] carve(T)(ref size_t offset) {
        auto slice = (cast(T*)(buf.ptr + offset))[0 .. MAX_N];
        offset += T.sizeof * MAX_N;
        return slice;
    }

    auto ch = new Chain1d;
    ch.pos      = carve!Vec3(off);
    ch.dirs     = carve!(Vec3[4])(off);
    ch.faceIdx  = carve!int(off);
    ch.ops      = carve!Site1dOps(off);
    assert(off == buf.length);

    return ch;
}

/// Compute tau at site n from the analytic exit direction.
Mat4 siteTau(const Chain1d* ch, int n) {
    return makeTau(helixExitDir(n));
}

/// Build chain geometry and operators up to and including site i.
void ensureSite(Chain1d* ch, int i) {
    assert(i >= 0 && i < MAX_N, "site out of range");
    while (ch.builtUpTo <= i) {
        int n = ch.builtUpTo;
        // Analytic geometry from vertex formula — no reflections needed.
        ch.pos[n] = helixCentroid(n);
        ch.dirs[n] = helixVertexDirs(n);
        ch.faceIdx[n] = ch.pat[n % 4];

        // tau and projectors (local — not stored per-site)
        Mat4 tau = siteTau(ch, n);
        Mat4 Pp = projPlus(tau);
        Mat4 Pm = projMinus(tau);

        // Shift blocks
        if (n > 0) {
            Mat4 tauPrev = siteTau(ch, n - 1);
            ch.ops[n].bwdBlock = mul(frameTransport(tau, tauPrev), Pm);
            Mat4 PpPrev = projPlus(tauPrev);
            ch.ops[n-1].fwdBlock = mul(frameTransport(tauPrev, tau), PpPrev);

            // U(2) gauge at site n-1: build the full 4×4 unitary gauge matrix
            // G = I + (e^{iψ}-1)(|b2+><b2+| + |b2-><b2-|)
            // using tauPrev's eigenbasis, then multiply both blocks by G.
            // The phase ψ depends on site position: gaugePhases[(n-1) % 4].
            double gphase = ch.gaugePhases[(n-1) % 4];
            if (gphase != 0.0) {
                Mat4 PmPrev = projMinus(tauPrev);

                // Gram-Schmidt b2 for P+ and P- of tauPrev
                double[4][2][2] bRe = 0, bIm = 0;  // [pm][basis_idx][component]
                // pm=0 → P+, pm=1 → P-
                Mat4[2] projs = [PpPrev, PmPrev];
                foreach (pm; 0 .. 2) {
                    int nf = 0;
                    foreach (col; 0 .. 4) {
                        if (nf >= 2) break;
                        double[4] vR = 0, vI = 0;
                        foreach (aa; 0 .. 4) { vR[aa] = projs[pm].re[4*aa+col]; vI[aa] = projs[pm].im[4*aa+col]; }
                        foreach (jj; 0 .. nf) {
                            double dr = 0, di = 0;
                            foreach (aa; 0 .. 4) { dr += bRe[pm][jj][aa]*vR[aa]+bIm[pm][jj][aa]*vI[aa]; di += bRe[pm][jj][aa]*vI[aa]-bIm[pm][jj][aa]*vR[aa]; }
                            foreach (aa; 0 .. 4) { vR[aa] -= dr*bRe[pm][jj][aa]-di*bIm[pm][jj][aa]; vI[aa] -= dr*bIm[pm][jj][aa]+di*bRe[pm][jj][aa]; }
                        }
                        double nm2 = 0;
                        foreach (aa; 0 .. 4) nm2 += vR[aa]*vR[aa]+vI[aa]*vI[aa];
                        if (nm2 > 1e-10) { double inv = 1.0/sqrt(nm2); foreach (aa; 0 .. 4) { bRe[pm][nf][aa]=inv*vR[aa]; bIm[pm][nf][aa]=inv*vI[aa]; } nf++; }
                    }
                }

                // Build G = I + c*(|b2+><b2+| + |b2-><b2-|) where c = e^{iψ}-1
                double cRe = cos(gphase) - 1.0;
                double cIm = sin(gphase);
                Mat4 G = Mat4.eye();
                foreach (aa; 0 .. 4)
                    foreach (bb; 0 .. 4) {
                        // |b2+><b2+| + |b2-><b2-| contribution
                        double pRe = 0, pIm = 0;
                        foreach (pm; 0 .. 2) {
                            // b2[aa] * conj(b2[bb])
                            double bRaa = bRe[pm][1][aa], bIaa = bIm[pm][1][aa];
                            double bRbb = bRe[pm][1][bb], bIbb = bIm[pm][1][bb];
                            pRe += bRaa*bRbb + bIaa*bIbb;
                            pIm += bIaa*bRbb - bRaa*bIbb;
                        }
                        // c * (|b2+><b2+| + |b2-><b2-|)
                        int idx = 4*aa+bb;
                        G.re[idx] += cRe*pRe - cIm*pIm;
                        G.im[idx] += cRe*pIm + cIm*pRe;
                    }

                // Apply: fwdBlock[n-1] = fwdBlock[n-1] * G
                ch.ops[n-1].fwdBlock = mul(ch.ops[n-1].fwdBlock, G);
                // Apply: bwdBlock[n-1] = bwdBlock[n-1] * G (if exists)
                if (n-1 > 0)
                    ch.ops[n-1].bwdBlock = mul(ch.ops[n-1].bwdBlock, G);
            }

            // (old gauge2PhaseM block removed — now handled by gaugePhases)
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

            ch.ops[n].coin1 = buildCoinMatrix([f1.x, f1.y, f1.z], ch.ct, ch.st);
            ch.ops[n].coin2 = buildCoinMatrix([f2.x, f2.y, f2.z], ch.ct, ch.st);
            ch.useCoin2 = true;

            if (ch.coinType == 4) {
                ch.ops[n].coin3 = buildBetaCoin(ch.ct, ch.st);
                ch.useCoin3 = true;
            } else if (ch.coinType == 5) {
                ch.ops[n].coin3 = buildCoinMatrix([e.x, e.y, e.z], ch.ct, ch.st);
                ch.useCoin3 = true;
            }
        } else if (ch.coinType == 1) {
            ch.ops[n].coin1 = buildCoinMatrix([e.x, e.y, e.z], ch.ct, ch.st);
        } else {
            ch.ops[n].coin1 = buildBetaCoin(ch.ct, ch.st);
        }

        // V-mixing
        if (ch.mixPhi != 0.0) {
            ch.ops[n].vmix = buildVmix(Pp, Pm, ch.mixPhi);
        } else {
            ch.ops[n].vmix = Mat4.eye();
        }

        ch.builtUpTo++;
    }
}

/// Build coin matrix: exp(-i theta (d . alpha)) = cos(theta)I - i sin(theta)(d . alpha)
Mat4 buildCoinMatrix(double[3] d, double ct, double st) {
    Mat4 m;
    foreach (a; 0 .. 4)
        foreach (b; 0 .. 4) {
            // (d . alpha)_{ab}
            double eaRe = 0, eaIm = 0;
            foreach (k; 0 .. 3) {
                auto al = alpha(k);
                eaRe += d[k] * al.re[4*a+b];
                eaIm += d[k] * al.im[4*a+b];
            }
            // C = cos(theta) delta - i sin(theta) (d.alpha)
            // -i*st * (eaRe + i*eaIm) = st*eaIm - i*st*eaRe
            m.re[4*a+b] = (a == b ? ct : 0.0) + st * eaIm;
            m.im[4*a+b] = -st * eaRe;
        }
    return m;
}

/// Build beta coin: exp(-i theta beta)
Mat4 buildBetaCoin(double ct, double st) {
    Mat4 m;
    double[4] bd = [1, 1, -1, -1];
    foreach (a; 0 .. 4) {
        m.re[4*a+a] = ct;
        m.im[4*a+a] = -st * bd[a];
    }
    return m;
}

/// Build V-mixing operator from projectors.
Mat4 buildVmix(Mat4 Pp, Mat4 Pm, double mixPhi) {
    double cp = cos(mixPhi), sp = sin(mixPhi);

    // Gram-Schmidt for P+ and P- bases
    double[2][4] ppBRe = 0, ppBIm = 0, pmBRe = 0, pmBIm = 0;
    int npFound = 0, nmFound = 0;
    foreach (col; 0 .. 4) {
        if (npFound >= 2 && nmFound >= 2) break;
        if (npFound < 2) {
            double[4] vRe = 0, vIm = 0;
            foreach (a; 0 .. 4) { vRe[a] = Pp.re[4*a+col]; vIm[a] = Pp.im[4*a+col]; }
            foreach (j; 0 .. npFound) {
                double dRe = 0, dIm = 0;
                foreach (a; 0 .. 4) {
                    dRe += ppBRe[a][j] * vRe[a] + ppBIm[a][j] * vIm[a];
                    dIm += ppBRe[a][j] * vIm[a] - ppBIm[a][j] * vRe[a];
                }
                foreach (a; 0 .. 4) {
                    vRe[a] -= dRe * ppBRe[a][j] - dIm * ppBIm[a][j];
                    vIm[a] -= dRe * ppBIm[a][j] + dIm * ppBRe[a][j];
                }
            }
            double nm = 0;
            foreach (a; 0 .. 4) nm += vRe[a] * vRe[a] + vIm[a] * vIm[a];
            if (nm > 1e-10) {
                double inv = 1.0 / sqrt(nm);
                foreach (a; 0 .. 4) { ppBRe[a][npFound] = inv * vRe[a]; ppBIm[a][npFound] = inv * vIm[a]; }
                npFound++;
            }
        }
        if (nmFound < 2) {
            double[4] vRe = 0, vIm = 0;
            foreach (a; 0 .. 4) { vRe[a] = Pm.re[4*a+col]; vIm[a] = Pm.im[4*a+col]; }
            foreach (j; 0 .. nmFound) {
                double dRe = 0, dIm = 0;
                foreach (a; 0 .. 4) {
                    dRe += pmBRe[a][j] * vRe[a] + pmBIm[a][j] * vIm[a];
                    dIm += pmBRe[a][j] * vIm[a] - pmBIm[a][j] * vRe[a];
                }
                foreach (a; 0 .. 4) {
                    vRe[a] -= dRe * pmBRe[a][j] - dIm * pmBIm[a][j];
                    vIm[a] -= dRe * pmBIm[a][j] + dIm * pmBRe[a][j];
                }
            }
            double nm = 0;
            foreach (a; 0 .. 4) nm += vRe[a] * vRe[a] + vIm[a] * vIm[a];
            if (nm > 1e-10) {
                double inv = 1.0 / sqrt(nm);
                foreach (a; 0 .. 4) { pmBRe[a][nmFound] = inv * vRe[a]; pmBIm[a][nmFound] = inv * vIm[a]; }
                nmFound++;
            }
        }
    }

    // M = sum_j |pm_j><pp_j| + |pp_j><pm_j|
    Mat4 V;
    foreach (j; 0 .. 2)
        foreach (a; 0 .. 4)
            foreach (b; 0 .. 4) {
                int ab = 4*a+b;
                V.re[ab] += pmBRe[a][j] * ppBRe[b][j] + pmBIm[a][j] * ppBIm[b][j];
                V.im[ab] += pmBIm[a][j] * ppBRe[b][j] - pmBRe[a][j] * ppBIm[b][j];
                V.re[ab] += ppBRe[a][j] * pmBRe[b][j] + ppBIm[a][j] * pmBIm[b][j];
                V.im[ab] += ppBIm[a][j] * pmBRe[b][j] - ppBRe[a][j] * pmBIm[b][j];
            }
    // V = cos(phi)I + i sin(phi) M
    foreach (a; 0 .. 4)
        foreach (b; 0 .. 4) {
            int ab = 4*a+b;
            double mRe = V.re[ab], mIm = V.im[ab];
            V.re[ab] = (a == b ? cp : 0.0) + sp * (-mIm);
            V.im[ab] = sp * mRe;
        }
    return V;
}

/// Apply a precomputed 4x4 matrix to spinor at site i, in place.
void applyMat4InPlace(Mat4 M, double[] psiRe, double[] psiIm, int i) {
    double[4] resRe = 0, resIm = 0;
    matVecSplit(M, &psiRe[4*i], &psiIm[4*i], resRe.ptr, resIm.ptr);
    psiRe[4*i .. 4*i+4] = resRe[];
    psiIm[4*i .. 4*i+4] = resIm[];
}

void run1d(Walk1dParams p) {
    double ct = cos(p.theta), st = sin(p.theta);

    int[4] pat = p.spiralType == 1
        ? [0, 2, 1, 3]   // L helix (perpendicular to R)
        : [1, 3, 0, 2];  // R helix

    stderr.writefln("=== walk_1d_d: theta=%.3f sigma=%.1f steps=%d coin=%d k0=%.4f spiral=%s ic=%d ===",
                    p.theta, p.sigma, p.nSteps, p.coinType, p.k0,
                    p.spiralType == 1 ? "L" : "R", p.icType);

    auto ch = newChain1d();
    ch.pat = pat;
    ch.ct = ct;
    ch.st = st;
    ch.coinType = p.coinType;
    ch.mixPhi = p.mixPhi;
    ch.gaugePhases = p.gaugePhases;

    // Initialize wavepacket centered at MAX_N/2
    int center = MAX_N / 2;
    int lo = center - cast(int)(4 * p.sigma) - 10;
    int hi = center + cast(int)(4 * p.sigma) + 10;
    if (lo < 0) lo = 0;
    if (hi >= MAX_N) hi = MAX_N - 1;
    ensureSite(ch, hi);

    auto psiRe = new double[4 * MAX_N];
    auto psiIm = new double[4 * MAX_N];
    psiRe[] = 0;
    psiIm[] = 0;

    // Build initial spinor at center site based on IC type
    double[4] chi0Re, chi0Im;
    chi0Re[] = 0; chi0Im[] = 0;

    if (p.icType == 0) {
        // (1,0,0,0)
        chi0Re = [1, 0, 0, 0];
    } else if (p.icType == 1) {
        // (1,0,1,0)/√2
        double inv = 1.0 / sqrt(2.0);
        chi0Re = [inv, 0, inv, 0];
    } else if (p.icType == 4) {
        // Walk-optimal spinor (from low-energy eigenstate construction)
        // Frame-transported by the Gaussian IC code below.
        chi0Re = [0.5354, 0.5058, 0.3043, 0.5959];
        chi0Im = [-0.0235, -0.0777, -0.0402, -0.0406];
        // Normalize
        double nChi = 0;
        foreach (a; 0 .. 4) nChi += chi0Re[a]*chi0Re[a] + chi0Im[a]*chi0Im[a];
        double invChi = 1.0 / sqrt(nChi);
        foreach (a; 0 .. 4) { chi0Re[a] *= invChi; chi0Im[a] *= invChi; }
    } else {
        // P+/P- symmetric: project (1,0,0,0) onto P+ and P-,
        // normalize each, sum to get equal weight in both subspaces.
        Mat4 tau0 = siteTau(ch, center);
        Mat4 Pp = projPlus(tau0);
        Mat4 Pm = projMinus(tau0);
        double[4] seed = [1, 0, 0, 0];
        double[4] seedIm = [0, 0, 0, 0];
        double[4] vpRe = 0, vpIm = 0, vmRe = 0, vmIm = 0;
        matVecSplit(Pp, seed.ptr, seedIm.ptr, vpRe.ptr, vpIm.ptr);
        matVecSplit(Pm, seed.ptr, seedIm.ptr, vmRe.ptr, vmIm.ptr);
        // Normalize each projection
        double npSq = 0, nmSq = 0;
        foreach (a; 0 .. 4) {
            npSq += vpRe[a]*vpRe[a] + vpIm[a]*vpIm[a];
            nmSq += vmRe[a]*vmRe[a] + vmIm[a]*vmIm[a];
        }
        double invP = 1.0 / sqrt(npSq), invM = 1.0 / sqrt(nmSq);
        foreach (a; 0 .. 4) {
            chi0Re[a] = invP * vpRe[a] + invM * vmRe[a];
            chi0Im[a] = invP * vpIm[a] + invM * vmIm[a];
        }
        // Normalize the sum
        double nChi = 0;
        foreach (a; 0 .. 4) nChi += chi0Re[a]*chi0Re[a] + chi0Im[a]*chi0Im[a];
        double invChi = 1.0 / sqrt(nChi);
        foreach (a; 0 .. 4) { chi0Re[a] *= invChi; chi0Im[a] *= invChi; }
    }

    stderr.writefln("IC type %d: chi0 = (%.4f%+.4fi, %.4f%+.4fi, %.4f%+.4fi, %.4f%+.4fi)",
                    p.icType,
                    chi0Re[0], chi0Im[0], chi0Re[1], chi0Im[1],
                    chi0Re[2], chi0Im[2], chi0Re[3], chi0Im[3]);

    // icType=3: load IC directly from /tmp/optimal_ic.dat
    if (p.icType == 3) {
        import std.stdio : File;
        import std.conv : to;
        import std.array : split;
        auto f = File("/tmp/optimal_ic.dat", "r");
        foreach (line; f.byLine()) {
            auto s = cast(string) line;
            if (s.length == 0 || s[0] == '#') continue;
            auto parts = s.split();
            if (parts.length < 9) continue;
            int offset = parts[0].to!int;
            int site = center + offset;
            if (site < 0 || site >= MAX_N) continue;
            ensureSite(ch, site);
            foreach (a; 0 .. 4) {
                psiRe[4*site + a] = parts[1 + 2*a].to!double;
                psiIm[4*site + a] = parts[2 + 2*a].to!double;
            }
        }
        // Normalize
        double norm2f = 0;
        foreach (i; lo .. hi + 1)
            foreach (a; 0 .. 4) {
                norm2f += psiRe[4*i+a]*psiRe[4*i+a] + psiIm[4*i+a]*psiIm[4*i+a];
            }
        double nff = 1.0 / sqrt(norm2f);
        foreach (i; lo .. hi + 1)
            foreach (a; 0 .. 4) {
                psiRe[4*i+a] *= nff;
                psiIm[4*i+a] *= nff;
            }
        stderr.writefln("Loaded IC from /tmp/optimal_ic.dat, norm=%.6f", norm2f);
    } else {
        // Gaussian IC with momentum kick.
        // Frame transport keeps spinor aligned with local tau eigenbasis.
        bool useTransport = true;
        double norm2 = 0;

        // Forward from center
        {
            double[4] chiCurRe = chi0Re;
            double[4] chiCurIm = chi0Im;
            foreach (i; center .. hi + 1) {
                double x = cast(double)(i - center);
                double w = exp(-x*x / (2 * p.sigma * p.sigma));
                double phRe = cos(p.k0 * x), phIm = sin(p.k0 * x);
                foreach (a; 0 .. 4) {
                    psiRe[4*i + a] = w * (phRe * chiCurRe[a] - phIm * chiCurIm[a]);
                    psiIm[4*i + a] = w * (phRe * chiCurIm[a] + phIm * chiCurRe[a]);
                }
                norm2 += w * w;
                if (useTransport && i < hi) {
                    ensureSite(ch, i + 1);
                    Mat4 U = frameTransport(siteTau(ch, i), siteTau(ch, i+1));
                    double[4] chiNextRe = 0, chiNextIm = 0;
                    matVecSplit(U, chiCurRe.ptr, chiCurIm.ptr, chiNextRe.ptr, chiNextIm.ptr);
                    chiCurRe = chiNextRe;
                    chiCurIm = chiNextIm;
                }
            }
        }

        // Backward from center
        {
            double[4] chiCurRe = chi0Re;
            double[4] chiCurIm = chi0Im;
            foreach_reverse (i; lo .. center) {
                if (useTransport) {
                    ensureSite(ch, i);
                    Mat4 U = frameTransport(siteTau(ch, i+1), siteTau(ch, i));
                    double[4] chiNextRe = 0, chiNextIm = 0;
                    matVecSplit(U, chiCurRe.ptr, chiCurIm.ptr, chiNextRe.ptr, chiNextIm.ptr);
                    chiCurRe = chiNextRe;
                    chiCurIm = chiNextIm;
                }
                double x = cast(double)(i - center);
                double w = exp(-x*x / (2 * p.sigma * p.sigma));
                double phRe = cos(p.k0 * x), phIm = sin(p.k0 * x);
                foreach (a; 0 .. 4) {
                    psiRe[4*i + a] = w * (phRe * chiCurRe[a] - phIm * chiCurIm[a]);
                    psiIm[4*i + a] = w * (phRe * chiCurIm[a] + phIm * chiCurRe[a]);
                }
                norm2 += w * w;
            }
        }

        double nf = 1.0 / sqrt(norm2);
        foreach (i; lo .. hi + 1)
            foreach (a; 0 .. 4) {
                psiRe[4*i + a] *= nf;
                psiIm[4*i + a] *= nf;
            }
    }
    int activeLo = lo, activeHi = hi + 1;
    immutable double ampThresh = 1e-15;

    stderr.writefln("Initial: active [%d, %d), center=%d, width=%d",
                    activeLo, activeHi, center, activeHi - activeLo);

    // Output header
    writefln("# theta=%.4f sigma=%.1f n_steps=%d coin=%d k0=%.6f",
             p.theta, p.sigma, p.nSteps, p.coinType, p.k0);
    writefln("# t norm x_mean x2 active_width");

    auto tmpRe = new double[4 * MAX_N];
    auto tmpIm = new double[4 * MAX_N];

    foreach (t; 0 .. p.nSteps + 1) {
        // Observables
        double totalProb = 0, mx = 0;
        foreach (i; activeLo .. activeHi) {
            double prob = 0;
            foreach (a; 0 .. 4) {
                double re = psiRe[4*i + a];
                double im = psiIm[4*i + a];
                prob += re * re + im * im;
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
                double re = psiRe[4*i + a];
                double im = psiIm[4*i + a];
                prob += re * re + im * im;
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
                applyMat4InPlace(ch.ops[i].coin1, psiRe, psiIm, i);
            if (ch.useCoin2)
                foreach (i; alo .. ahi)
                    applyMat4InPlace(ch.ops[i].coin2, psiRe, psiIm, i);
            if (ch.useCoin3)
                foreach (i; alo .. ahi)
                    applyMat4InPlace(ch.ops[i].coin3, psiRe, psiIm, i);

            // Shift
            if (activeLo > 0) ensureSite(ch, activeLo - 1);
            ensureSite(ch, activeHi);

            int zlo = (activeLo > 0) ? activeLo - 1 : 0;
            int zhi = activeHi + 1;
            tmpRe[4*zlo .. 4*zhi] = 0;
            tmpIm[4*zlo .. 4*zhi] = 0;

            foreach (i; zlo .. zhi) {
                double[4] accRe = 0, accIm = 0;

                if (i-1 >= alo && i-1 < ahi) {
                    matVecSplit(ch.ops[i-1].fwdBlock,
                                &psiRe[4*(i-1)], &psiIm[4*(i-1)],
                                accRe.ptr, accIm.ptr);
                }

                if (i+1 >= alo && i+1 < ahi) {
                    double[4] bRe = 0, bIm = 0;
                    matVecSplit(ch.ops[i+1].bwdBlock,
                                &psiRe[4*(i+1)], &psiIm[4*(i+1)],
                                bRe.ptr, bIm.ptr);
                    foreach (a; 0 .. 4) {
                        accRe[a] += bRe[a];
                        accIm[a] += bIm[a];
                    }
                }

                tmpRe[4*i .. 4*i+4] = accRe[];
                tmpIm[4*i .. 4*i+4] = accIm[];
            }

            // V-mixing
            if (p.mixPhi != 0.0)
                foreach (i; zlo .. zhi)
                    applyMat4InPlace(ch.ops[i].vmix, tmpRe, tmpIm, i);

            // Update active region
            if (activeLo > 0) {
                double prob = 0;
                foreach (a; 0 .. 4) {
                    double re = tmpRe[4*(activeLo-1) + a];
                    double im = tmpIm[4*(activeLo-1) + a];
                    prob += re * re + im * im;
                }
                if (prob > ampThresh) activeLo--;
            }
            {
                double prob = 0;
                foreach (a; 0 .. 4) {
                    double re = tmpRe[4*activeHi + a];
                    double im = tmpIm[4*activeHi + a];
                    prob += re * re + im * im;
                }
                if (prob > ampThresh) activeHi++;
            }

            // Copy tmp to psi
            psiRe[4*activeLo .. 4*activeHi] = tmpRe[4*activeLo .. 4*activeHi];
            psiIm[4*activeLo .. 4*activeHi] = tmpIm[4*activeLo .. 4*activeHi];
        }
    }

    // Output density
    {
        auto pf = File("/tmp/walk_1d_density.dat", "w");
        double total = 0;
        foreach (i; activeLo .. activeHi)
            foreach (a; 0 .. 4) {
                double re = psiRe[4*i + a];
                double im = psiIm[4*i + a];
                total += re * re + im * im;
            }
        pf.writefln("# site position prob prob_plus prob_minus");
        foreach (i; activeLo .. activeHi) {
            double prob = 0;
            foreach (a; 0 .. 4) {
                double re = psiRe[4*i + a];
                double im = psiIm[4*i + a];
                prob += re * re + im * im;
            }
            // P+ and P- components
            double pp = 0, pm = 0;
            Mat4 tau = siteTau(ch, i);
            foreach (a; 0 .. 4) {
                double spRe = 0, spIm = 0, smRe = 0, smIm = 0;
                foreach (b; 0 .. 4) {
                    double tRe = tau.re[4*a+b];
                    double tIm = tau.im[4*a+b];
                    double dab = (a == b) ? 1.0 : 0.0;
                    double pRe = psiRe[4*i + b];
                    double pIm = psiIm[4*i + b];
                    // (dab + tau) * psi
                    double pplusRe = (dab + tRe);
                    double pplusIm = tIm;
                    spRe += pplusRe * pRe - pplusIm * pIm;
                    spIm += pplusRe * pIm + pplusIm * pRe;
                    // (dab - tau) * psi
                    double pminusRe = (dab - tRe);
                    double pminusIm = -tIm;
                    smRe += pminusRe * pRe - pminusIm * pIm;
                    smIm += pminusRe * pIm + pminusIm * pRe;
                }
                pp += spRe * spRe + spIm * spIm;
                pm += smRe * smRe + smIm * smIm;
            }
            pp *= 0.25; pm *= 0.25;
            double r = norm(ch.pos[i]);
            if (i < center) r = -r;
            pf.writefln("%d %.6f %.10e %.10e %.10e",
                        i - center, r, prob / total, pp / total, pm / total);
        }
        stderr.writefln("Density -> /tmp/walk_1d_density.dat (%d sites)", activeHi - activeLo);
    }

    // ch is GC-allocated; no manual free needed.
}

version(walk_1d_exe) {
    int main(string[] args) {
        auto p = parseArgs1d(args);
        run1d(p);
        return 0;
    }
}
