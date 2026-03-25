/**
 * operators.d — Walk step operators: shift, coin, V-mixing, and pruning.
 *
 * Templated on hasCoin: when false, coin application is compiled out entirely.
 * All operators use precomputed per-site blocks from the chain deques.
 */
module operators;

import std.math : sqrt, fabs, cos, sin, exp;
import geometry : Vec3, dot, norm, helixStep, reorth;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, matVecSplit, alpha;
import lattice : Lattice, nextFace, prevFace, PAT_R, PAT_L;

// ---- Spinor helpers ----

double spinorNorm2(bool hasCoin)(const Lattice!hasCoin lat, int s) {
    double n = 0;
    foreach (a; 0 .. 4) {
        double re = lat.psiRe[4*s + a];
        double im = lat.psiIm[4*s + a];
        n += re * re + im * im;
    }
    return n;
}

// ---- Adaptive shift operator ----

struct ShiftResult {
    int nCreated = 0;
    double probAbsorbed = 0;
}

private int tryExtendFwd(bool hasCoin)(ref Lattice!hasCoin lat, int s, bool isR,
                         const int[4] pat, ref double[4] shiftedRe,
                         ref double[4] shiftedIm, double thresh2) {
    double amp2 = 0;
    foreach (a; 0 .. 4) amp2 += shiftedRe[a]*shiftedRe[a] + shiftedIm[a]*shiftedIm[a];
    if (amp2 < thresh2) return -1;

    int face = lat.chainFace(s, isR);
    int nf = nextFace(pat, face);

    Vec3 p = lat.sites[s].pos;
    auto d = lat.sites[s].dirs;
    helixStep(p, d, face);
    reorth(d);

    int nb = lat.allocSite(p, d);
    lat.setChainFace(nb, isR, nf);
    int chainId = isR ? lat.sites[s].rChain : lat.sites[s].lChain;
    lat.chainAppend(chainId, nb);
    return nb;
}

private int tryExtendBwd(bool hasCoin)(ref Lattice!hasCoin lat, int s, bool isR,
                         const int[4] pat, ref double[4] shiftedRe,
                         ref double[4] shiftedIm, double thresh2) {
    double amp2 = 0;
    foreach (a; 0 .. 4) amp2 += shiftedRe[a]*shiftedRe[a] + shiftedIm[a]*shiftedIm[a];
    if (amp2 < thresh2) return -1;

    int face = lat.chainFace(s, isR);
    int pf = prevFace(pat, face);

    Vec3 p = lat.sites[s].pos;
    auto d = lat.sites[s].dirs;
    helixStep(p, d, pf);
    reorth(d);

    int nb = lat.allocSite(p, d);
    lat.setChainFace(nb, isR, pf);
    int chainId = isR ? lat.sites[s].rChain : lat.sites[s].lChain;
    lat.chainPrepend(chainId, nb);
    return nb;
}

ShiftResult applyShift(bool hasCoin)(ref Lattice!hasCoin lat, bool isR,
                                     const int[4] pat, double thresh2) {
    import core.thread : Thread;
    import std.parallelism : parallel, taskPool;
    alias ChainT = Lattice!hasCoin.ChainT;
    alias Ops = Lattice!hasCoin.Ops;

    int ns = lat.nsites;
    ShiftResult result;

    // Zero tmp
    lat.tmpRe[0 .. 4 * ns] = 0;
    lat.tmpIm[0 .. 4 * ns] = 0;

    // Copy identity for sites not on a chain of this type
    foreach (s; 0 .. ns) {
        if (lat.chainFace(s, isR) < 0) {
            lat.tmpRe[4*s .. 4*s+4] = lat.psiRe[4*s .. 4*s+4];
            lat.tmpIm[4*s .. 4*s+4] = lat.psiIm[4*s .. 4*s+4];
        }
    }

    // Process chains in two passes:
    // 1. Interior sites (parallel) — gather from neighbors, no write conflicts
    // 2. Chain-end extension (serial) — mutates lattice

    // Collect chain indices for this chirality (reuse static buffer)
    static int[] chainIds;
    if (chainIds.length < lat.chains.length)
        chainIds = new int[lat.chains.length];
    int nci = 0;
    foreach (ci, ref ch; lat.chains)
        if (ch.isR == isR && ch.ops.length > 0)
            chainIds[nci++] = cast(int) ci;
    auto activeChains = chainIds[0 .. nci];

    // Pass 1: interior gather (parallel-safe: each site writes only to its own tmp)
    import std.parallelism : parallel;
    foreach (ci; parallel(activeChains)) {
        auto ch = &lat.chains[ci];
        int n = ch.ops.length;

        foreach (i; 0 .. n) {
            int s = ch.ops[i].siteId;
            double[4] accRe = 0, accIm = 0;

            if (i > 0) {
                int prev = ch.ops[i-1].siteId;
                matVecSplit(ch.ops[i-1].fwdBlock,
                            &lat.psiRe[4*prev], &lat.psiIm[4*prev],
                            accRe.ptr, accIm.ptr);
            }

            if (i < n - 1) {
                int next = ch.ops[i+1].siteId;
                double[4] bRe = 0, bIm = 0;
                matVecSplit(ch.ops[i+1].bwdBlock,
                            &lat.psiRe[4*next], &lat.psiIm[4*next],
                            bRe.ptr, bIm.ptr);
                foreach (a; 0 .. 4) {
                    accRe[a] += bRe[a];
                    accIm[a] += bIm[a];
                }
            }

            foreach (a; 0 .. 4) {
                lat.tmpRe[4*s + a] += accRe[a];
                lat.tmpIm[4*s + a] += accIm[a];
            }
        }
    }

    // Pass 2: chain-end extension (serial — mutates lattice)
    foreach (ci; activeChains) {
        auto ch = &lat.chains[ci];
        int n = ch.ops.length;

        // Forward end
        {
            int endSite = ch.ops[n-1].siteId;
            int face = lat.chainFace(endSite, isR);
            Mat4 tau = makeTau(lat.sites[endSite].dirs[face]);
            Mat4 Pp = projPlus(tau);
            double[4] shRe = 0, shIm = 0;
            matVecSplit(Pp, &lat.psiRe[4*endSite], &lat.psiIm[4*endSite],
                        shRe.ptr, shIm.ptr);
            int nb = tryExtendFwd!hasCoin(lat, endSite, isR, pat, shRe, shIm, thresh2);
            if (nb >= 0) {
                auto newOp = lat.siteOps(endSite, isR);
                double[4] resRe = 0, resIm = 0;
                matVecSplit(newOp.fwdBlock, &lat.psiRe[4*endSite], &lat.psiIm[4*endSite],
                            resRe.ptr, resIm.ptr);
                foreach (a; 0 .. 4) {
                    lat.tmpRe[4*nb + a] += resRe[a];
                    lat.tmpIm[4*nb + a] += resIm[a];
                }
                result.nCreated++;
            } else if (nb == -1) {
                foreach (a; 0 .. 4)
                    result.probAbsorbed += shRe[a]*shRe[a] + shIm[a]*shIm[a];
            }
        }

        // Backward end
        {
            int endSite = ch.ops[0].siteId;
            int face = lat.chainFace(endSite, isR);
            Mat4 tau = makeTau(lat.sites[endSite].dirs[face]);
            Mat4 Pm = projMinus(tau);
            double[4] shRe = 0, shIm = 0;
            matVecSplit(Pm, &lat.psiRe[4*endSite], &lat.psiIm[4*endSite],
                        shRe.ptr, shIm.ptr);
            int nb = tryExtendBwd!hasCoin(lat, endSite, isR, pat, shRe, shIm, thresh2);
            if (nb >= 0) {
                auto newOp = lat.siteOps(endSite, isR);
                double[4] resRe = 0, resIm = 0;
                matVecSplit(newOp.bwdBlock, &lat.psiRe[4*endSite], &lat.psiIm[4*endSite],
                            resRe.ptr, resIm.ptr);
                foreach (a; 0 .. 4) {
                    lat.tmpRe[4*nb + a] += resRe[a];
                    lat.tmpIm[4*nb + a] += resIm[a];
                }
                result.nCreated++;
            } else if (nb == -1) {
                foreach (a; 0 .. 4)
                    result.probAbsorbed += shRe[a]*shRe[a] + shIm[a]*shIm[a];
            }
        }
    }

    lat.swapBuffers();
    return result;
}

// ---- Coin operator ----

void applyCoin(bool hasCoin)(ref Lattice!hasCoin lat, bool isR) {
    static if (!hasCoin) return;
    else {
        int ns = lat.nsites;
        foreach (n; 0 .. ns) {
            auto op = lat.siteOps(n, isR);
            if (op is null) continue;

            // Apply coin1 then coin2 (dual parity)
            double[4] resRe = 0, resIm = 0;
            matVecSplit(op.coin1, &lat.psiRe[4*n], &lat.psiIm[4*n], resRe.ptr, resIm.ptr);
            lat.psiRe[4*n .. 4*n+4] = resRe[];
            lat.psiIm[4*n .. 4*n+4] = resIm[];

            resRe[] = 0; resIm[] = 0;
            matVecSplit(op.coin2, &lat.psiRe[4*n], &lat.psiIm[4*n], resRe.ptr, resIm.ptr);
            lat.psiRe[4*n .. 4*n+4] = resRe[];
            lat.psiIm[4*n .. 4*n+4] = resIm[];
        }
    }
}

// ---- V mixing ----

void applyVmix(bool hasCoin)(ref Lattice!hasCoin lat, bool isR, double mixPhi) {
    if (mixPhi == 0.0) return;
    double cp = cos(mixPhi), sp = sin(mixPhi);
    int ns = lat.nsites;

    foreach (s; 0 .. ns) {
        int face = lat.chainFace(s, isR);
        if (face < 0) continue;

        Mat4 tau = makeTau(lat.sites[s].dirs[face]);
        Mat4 Pp = projPlus(tau);
        Mat4 Pm = projMinus(tau);

        // Gram-Schmidt for P+ and P- bases
        double[4][2] ppBRe = 0, ppBIm = 0, pmBRe = 0, pmBIm = 0;
        int npFound = 0, nmFound = 0;
        foreach (col; 0 .. 4) {
            if (npFound >= 2 && nmFound >= 2) break;
            if (npFound < 2) {
                double[4] vRe = void, vIm = void;
                foreach (a; 0 .. 4) { vRe[a] = Pp.re[4*a+col]; vIm[a] = Pp.im[4*a+col]; }
                foreach (j; 0 .. npFound) {
                    double dRe = 0, dIm = 0;
                    foreach (a; 0 .. 4) {
                        dRe += ppBRe[a][j]*vRe[a] + ppBIm[a][j]*vIm[a];
                        dIm += ppBRe[a][j]*vIm[a] - ppBIm[a][j]*vRe[a];
                    }
                    foreach (a; 0 .. 4) {
                        vRe[a] -= dRe*ppBRe[a][j] - dIm*ppBIm[a][j];
                        vIm[a] -= dRe*ppBIm[a][j] + dIm*ppBRe[a][j];
                    }
                }
                double nm = 0;
                foreach (a; 0 .. 4) nm += vRe[a]*vRe[a] + vIm[a]*vIm[a];
                if (nm > 1e-10) {
                    double inv = 1.0 / sqrt(nm);
                    foreach (a; 0 .. 4) { ppBRe[a][npFound]=inv*vRe[a]; ppBIm[a][npFound]=inv*vIm[a]; }
                    npFound++;
                }
            }
            if (nmFound < 2) {
                double[4] vRe = void, vIm = void;
                foreach (a; 0 .. 4) { vRe[a] = Pm.re[4*a+col]; vIm[a] = Pm.im[4*a+col]; }
                foreach (j; 0 .. nmFound) {
                    double dRe = 0, dIm = 0;
                    foreach (a; 0 .. 4) {
                        dRe += pmBRe[a][j]*vRe[a] + pmBIm[a][j]*vIm[a];
                        dIm += pmBRe[a][j]*vIm[a] - pmBIm[a][j]*vRe[a];
                    }
                    foreach (a; 0 .. 4) {
                        vRe[a] -= dRe*pmBRe[a][j] - dIm*pmBIm[a][j];
                        vIm[a] -= dRe*pmBIm[a][j] + dIm*pmBRe[a][j];
                    }
                }
                double nm = 0;
                foreach (a; 0 .. 4) nm += vRe[a]*vRe[a] + vIm[a]*vIm[a];
                if (nm > 1e-10) {
                    double inv = 1.0 / sqrt(nm);
                    foreach (a; 0 .. 4) { pmBRe[a][nmFound]=inv*vRe[a]; pmBIm[a][nmFound]=inv*vIm[a]; }
                    nmFound++;
                }
            }
        }

        Mat4 V;
        foreach (j; 0 .. 2)
            foreach (a; 0 .. 4)
                foreach (b; 0 .. 4) {
                    int ab = 4*a+b;
                    V.re[ab] += pmBRe[a][j]*ppBRe[b][j] + pmBIm[a][j]*ppBIm[b][j];
                    V.im[ab] += pmBIm[a][j]*ppBRe[b][j] - pmBRe[a][j]*ppBIm[b][j];
                    V.re[ab] += ppBRe[a][j]*pmBRe[b][j] + ppBIm[a][j]*pmBIm[b][j];
                    V.im[ab] += ppBIm[a][j]*pmBRe[b][j] - ppBRe[a][j]*pmBIm[b][j];
                }
        foreach (a; 0 .. 4)
            foreach (b; 0 .. 4) {
                int ab = 4*a+b;
                double mRe = V.re[ab], mIm = V.im[ab];
                V.re[ab] = (a == b ? cp : 0.0) + sp * (-mIm);
                V.im[ab] = sp * mRe;
            }

        double[4] resRe = 0, resIm = 0;
        matVecSplit(V, &lat.psiRe[4*s], &lat.psiIm[4*s], resRe.ptr, resIm.ptr);
        lat.psiRe[4*s .. 4*s+4] = resRe[];
        lat.psiIm[4*s .. 4*s+4] = resIm[];
    }
}

// ---- Chain-end pruning ----

private bool checkPruneEligible(bool hasCoin)(const Lattice!hasCoin lat, int s,
                                              bool isR, bool isFwd, double thresh2) {
    double amp2 = spinorNorm2!hasCoin(lat, s);
    if (amp2 >= thresh2) return false;

    int nb = isFwd ? lat.chainPrev(s, isR) : lat.chainNext(s, isR);
    if (nb < 0) return true;

    int face = lat.chainFace(nb, isR);
    if (face < 0) return true;

    Mat4 tau = makeTau(lat.sites[nb].dirs[face]);
    Mat4 P = isFwd ? projPlus(tau) : projMinus(tau);

    double[4] vRe = 0, vIm = 0;
    matVecSplit(P, &lat.psiRe[4*nb], &lat.psiIm[4*nb], vRe.ptr, vIm.ptr);
    double flow2 = 0;
    foreach (a; 0 .. 4)
        flow2 += vRe[a]*vRe[a] + vIm[a]*vIm[a];
    return flow2 < thresh2;
}

private void unlinkChainEnd(bool hasCoin)(ref Lattice!hasCoin lat, int s,
                                          bool isR, bool isFwd) {
    int chainId = isR ? lat.sites[s].rChain : lat.sites[s].lChain;
    if (chainId < 0) return;

    auto ch = &lat.chains[chainId];
    if (isFwd) {
        assert(ch.ops[ch.ops.length - 1].siteId == s);
        ch.ops.popBack();
    } else {
        assert(ch.ops[0].siteId == s);
        ch.ops.popFront();
        ch.rootIdx--;
        for (int i = 0; i < ch.ops.length; i++) {
            int id = ch.ops[i].siteId;
            if (isR) lat.sites[id].rIdx--;
            else     lat.sites[id].lIdx--;
        }
    }

    if (isR) { lat.sites[s].rChain = -1; lat.sites[s].rIdx = -1; lat.sites[s].rFace = -1; }
    else     { lat.sites[s].lChain = -1; lat.sites[s].lIdx = -1; lat.sites[s].lFace = -1; }

    if (lat.sites[s].rChain < 0 && lat.sites[s].lChain < 0)
        lat.removeSite(s);
}

struct PruneResult { int count; double probPruned; }

PruneResult pruneChainEnds(bool hasCoin)(ref Lattice!hasCoin lat, double thresh2) {
    PruneResult total;
    int prunedThisPass;

    do {
        prunedThisPass = 0;
        foreach (s; 0 .. lat.nsites) {
            if (lat.chainFace(s, true) < 0 && lat.chainFace(s, false) < 0) continue;

            if (lat.chainFace(s, true) >= 0 && lat.chainNext(s, true) == -1) {
                if (checkPruneEligible!hasCoin(lat, s, true, true, thresh2)) {
                    total.probPruned += spinorNorm2!hasCoin(lat, s);
                    unlinkChainEnd!hasCoin(lat, s, true, true);
                    prunedThisPass++; continue;
                }
            }
            if (lat.chainFace(s, true) >= 0 && lat.chainPrev(s, true) == -1) {
                if (checkPruneEligible!hasCoin(lat, s, true, false, thresh2)) {
                    total.probPruned += spinorNorm2!hasCoin(lat, s);
                    unlinkChainEnd!hasCoin(lat, s, true, false);
                    prunedThisPass++; continue;
                }
            }
            if (lat.chainFace(s, false) >= 0 && lat.chainNext(s, false) == -1) {
                if (checkPruneEligible!hasCoin(lat, s, false, true, thresh2)) {
                    total.probPruned += spinorNorm2!hasCoin(lat, s);
                    unlinkChainEnd!hasCoin(lat, s, false, true);
                    prunedThisPass++; continue;
                }
            }
            if (lat.chainFace(s, false) >= 0 && lat.chainPrev(s, false) == -1) {
                if (checkPruneEligible!hasCoin(lat, s, false, false, thresh2)) {
                    total.probPruned += spinorNorm2!hasCoin(lat, s);
                    unlinkChainEnd!hasCoin(lat, s, false, false);
                    prunedThisPass++; continue;
                }
            }
        }
        total.count += prunedThisPass;
    } while (prunedThisPass > 0);

    return total;
}

// ---- D unit tests ----

unittest {
    import lattice : Lattice, DensityGrid, generateSites;
    import geometry : initTet;

    auto lat = Lattice!false.create(100000);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    double norm2 = 0;
    foreach (n; 0 .. lat.nsites) {
        double r2 = dot(lat.sites[n].pos, lat.sites[n].pos);
        double w = exp(-r2 / (2 * sigma * sigma));
        lat.psiRe[4*n] = w;
        norm2 += w * w;
    }
    double normFactor = 1.0 / sqrt(norm2);
    foreach (i; 0 .. 4 * lat.nsites) {
        lat.psiRe[i] *= normFactor;
        lat.psiIm[i] *= normFactor;
    }

    auto res = applyShift(lat, false, PAT_L, 1e-20);

    double postNorm = 0;
    foreach (i; 0 .. 4 * lat.nsites)
        postNorm += lat.psiRe[i]*lat.psiRe[i] + lat.psiIm[i]*lat.psiIm[i];
    double total = postNorm + res.probAbsorbed;
    assert(total > 0.95 && total < 1.05, "Norm + absorbed out of range");
}

unittest {
    import lattice : Lattice, DensityGrid, generateSites;

    auto lat = Lattice!true.create(100000);
    lat.coinCt = cos(0.5);
    lat.coinSt = sin(0.5);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    double val = 1.0 / sqrt(cast(double) lat.nsites);
    foreach (n; 0 .. lat.nsites)
        lat.psiRe[4*n] = val;

    double normBefore = 0;
    foreach (i; 0 .. 4 * lat.nsites)
        normBefore += lat.psiRe[i]*lat.psiRe[i] + lat.psiIm[i]*lat.psiIm[i];

    applyCoin(lat, true);

    double normAfter = 0;
    foreach (i; 0 .. 4 * lat.nsites)
        normAfter += lat.psiRe[i]*lat.psiRe[i] + lat.psiIm[i]*lat.psiIm[i];

    assert(fabs(normAfter - normBefore) < 1e-10, "Coin should preserve norm");
}
