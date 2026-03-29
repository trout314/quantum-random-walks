/**
 * operators.d — Walk step operators: shift, coin, V-mixing, and pruning.
 *
 * The shift is split into a pure linear-algebra step (pureShift) that
 * reports overflow at chain ends, and a separate handleOverflow that
 * decides whether to extend, absorb, or ignore.  This decouples the
 * unitary operator from lattice mutation and makes it reusable for both
 * the 1D and 3D walks.
 *
 * Templated on hasCoin: when false, coin application is compiled out.
 */
module operators;

import std.math : sqrt, fabs, cos, sin, exp;
import geometry : Vec3, dot, norm, chainCentroid;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, matVecSplit, alpha;
import lattice : Lattice, IS_R, IS_L, isRamLow;
import coarse_grid : CoarseGrid;

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

// ===========================================================================
//  Pure shift — no lattice mutation, returns overflow at chain ends
// ===========================================================================

/// Overflow from one chain end during the pure shift.
struct ChainEndOverflow {
    int chainIdx;       /// index into activeChains (for caller bookkeeping)
    int chainId;        /// lattice chain ID
    int endSiteId;      /// the boundary site whose P± projected out
    bool isFwd;         /// true = forward end (P+), false = backward (P-)
    double[4] ampRe = 0;
    double[4] ampIm = 0;
    Vec3 exitDir;       /// exit direction at the boundary site (for frame transport)
}

struct PureShiftResult {
    ChainEndOverflow[] overflows;
    int[] activeChains;    /// chain IDs that were processed
}

/**
 * Pure shift operator — gathers interior amplitudes and reports overflow.
 *
 * After this call:
 *   - lat.psiRe/Im contain the shifted amplitudes for interior sites
 *     and identity-copied amplitudes for sites not on this chain type.
 *   - Chain-end P± components are returned as overflow, NOT written to psi.
 *   - The caller must handle overflow (extend, absorb, or ignore).
 *
 * This function calls swapBuffers internally.
 */
PureShiftResult pureShift(bool hasCoin)(ref Lattice!hasCoin lat, bool isR) {
    import std.parallelism : parallel;
    alias ChainT = Lattice!hasCoin.ChainT;

    int ns = lat.nsites;
    PureShiftResult result;

    // Zero tmp
    lat.tmpRe[0 .. 4 * ns] = 0;
    lat.tmpIm[0 .. 4 * ns] = 0;

    // Copy identity for sites not on a chain of this type
    foreach (s; 0 .. ns) {
        if (!lat.hasChain(s, isR)) {
            lat.tmpRe[4*s .. 4*s+4] = lat.psiRe[4*s .. 4*s+4];
            lat.tmpIm[4*s .. 4*s+4] = lat.psiIm[4*s .. 4*s+4];
        }
    }

    // Collect active chains for this chirality
    static int[] chainIds;
    if (chainIds.length < lat.chains.length)
        chainIds = new int[lat.chains.length];
    int nci = 0;
    foreach (ci, ref ch; lat.chains)
        if (ch.isR == isR && ch.ops.length > 0)
            chainIds[nci++] = cast(int) ci;
    result.activeChains = chainIds[0 .. nci].dup;

    // ---- Interior gather (parallel across chains) ----
    enum BATCH_DIVISOR = 16;
    int batchSize = (nci > BATCH_DIVISOR) ? nci / BATCH_DIVISOR : 1;
    foreach (ci; parallel(result.activeChains, batchSize)) {
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

    // ---- Compute overflow at chain ends (serial, read-only) ----
    // Pre-allocate: at most 2 overflows per chain (fwd + bwd)
    result.overflows.reserve(2 * nci);

    foreach (idx, ci; result.activeChains) {
        auto ch = &lat.chains[ci];
        int n = ch.ops.length;

        // Forward end: P+ @ psi[last]
        {
            int endSite = ch.ops[n-1].siteId;
            Vec3 exitDir = lat.exitDirForSite(endSite, isR);
            Mat4 tau = makeTau(exitDir);
            Mat4 Pp = projPlus(tau);
            ChainEndOverflow ov;
            ov.chainIdx = cast(int) idx;
            ov.chainId = ci;
            ov.endSiteId = endSite;
            ov.isFwd = true;
            ov.exitDir = exitDir;
            matVecSplit(Pp, &lat.psiRe[4*endSite], &lat.psiIm[4*endSite],
                        ov.ampRe.ptr, ov.ampIm.ptr);
            result.overflows ~= ov;
        }

        // Backward end: P- @ psi[first]
        {
            int endSite = ch.ops[0].siteId;
            Vec3 exitDir = lat.exitDirForSite(endSite, isR);
            Mat4 tau = makeTau(exitDir);
            Mat4 Pm = projMinus(tau);
            ChainEndOverflow ov;
            ov.chainIdx = cast(int) idx;
            ov.chainId = ci;
            ov.endSiteId = endSite;
            ov.isFwd = false;
            ov.exitDir = exitDir;
            matVecSplit(Pm, &lat.psiRe[4*endSite], &lat.psiIm[4*endSite],
                        ov.ampRe.ptr, ov.ampIm.ptr);
            result.overflows ~= ov;
        }
    }

    lat.swapBuffers();

    return result;
}

// ===========================================================================
//  Overflow handling — the caller decides what to do with chain-end overflow
// ===========================================================================

struct ShiftResult {
    int nCreated = 0;
    int nCapFull = 0;
    double probAbsorbed = 0;
    int nPruned = 0;
    double probPruned = 0;
}

/**
 * Handle overflow from pureShift by extending chains or absorbing.
 *
 * For each overflow:
 *   - If |amp|² >= extThresh2: create a new site, frame-transport the
 *     overflow there, and deposit the transported amplitude in psi.
 *   - Otherwise: add |amp|² to probAbsorbed (and optionally to the
 *     coarse grid).
 *
 * This is the only function that mutates the lattice during a walk step.
 */
ShiftResult handleOverflow(bool hasCoin)(
    ref Lattice!hasCoin lat,
    bool isR,
    const(ChainEndOverflow)[] overflows,
    double extThresh2,
    CoarseGrid* cgrid = null)
{
    ShiftResult result;
    bool ramLow = false;

    foreach (ref ov; overflows) {
        double amp2 = 0;
        foreach (a; 0 .. 4)
            amp2 += ov.ampRe[a]*ov.ampRe[a] + ov.ampIm[a]*ov.ampIm[a];

        if (amp2 < extThresh2) {
            // Below threshold: absorb
            result.probAbsorbed += amp2;
            if (cgrid !is null)
                cgrid.addAmplitude(lat.sites[ov.endSiteId].pos,
                                   ov.ampRe.ptr, ov.ampIm.ptr, ov.exitDir);
            continue;
        }

        // Try to extend the chain
        int nb;
        if (ov.isFwd)
            nb = ramLow ? -1 : extendChainFwd!hasCoin(lat, ov.endSiteId, isR);
        else
            nb = ramLow ? -1 : extendChainBwd!hasCoin(lat, ov.endSiteId, isR);

        if (nb >= 0) {
            // Frame-transport the overflow amplitude to the new site
            Mat4 tauEnd = makeTau(ov.exitDir);
            Vec3 exitDirNb = lat.exitDirForSite(nb, isR);
            Mat4 tauNb = makeTau(exitDirNb);
            Mat4 U = frameTransport(tauEnd, tauNb);
            Mat4 proj = ov.isFwd ? projPlus(tauEnd) : projMinus(tauEnd);
            Mat4 block = mul(U, proj);

            // Apply block to the ORIGINAL psi at the end site.
            // After swapBuffers, psi holds the shifted values at interior sites.
            // But the overflow was computed from the PRE-swap psi.  We stored
            // the projected amplitude (P± @ old_psi) in ov.amp.  We just need
            // to transport it: new_amp = U @ ov.amp.
            double[4] resRe = 0, resIm = 0;
            foreach (a; 0 .. 4) {
                foreach (b; 0 .. 4) {
                    int ab = 4*a+b;
                    resRe[a] += U.re[ab]*ov.ampRe[b] - U.im[ab]*ov.ampIm[b];
                    resIm[a] += U.re[ab]*ov.ampIm[b] + U.im[ab]*ov.ampRe[b];
                }
            }

            lat.psiRe[4*nb .. 4*nb+4] = resRe[];
            lat.psiIm[4*nb .. 4*nb+4] = resIm[];
            result.nCreated++;
            if ((result.nCreated & 0x3FF) == 0 && isRamLow()) ramLow = true;
        } else {
            // Extension failed (lattice full or RAM low): absorb
            result.probAbsorbed += amp2;
            if (cgrid !is null)
                cgrid.addAmplitude(lat.sites[ov.endSiteId].pos,
                                   ov.ampRe.ptr, ov.ampIm.ptr, ov.exitDir);
        }
    }

    return result;
}

/// Extend a chain forward by one site. Returns new site ID, or -1 if full.
private int extendChainFwd(bool hasCoin)(ref Lattice!hasCoin lat, int endSite, bool isR) {
    int chainId = isR ? lat.sites[endSite].rChain : lat.sites[endSite].lChain;
    int nOps = lat.chains[chainId].ops.length;
    int newChainIdx = lat.chains[chainId].rootIdx + nOps;
    Vec3 p = chainCentroid(&lat.chains[chainId].origin, newChainIdx);

    int nb = lat.allocSite(p);
    if (nb < 0) return -1;
    lat.chainAppend(chainId, nb);
    lat.makeCrossChain(nb, !isR);
    return nb;
}

/// Extend a chain backward by one site. Returns new site ID, or -1 if full.
private int extendChainBwd(bool hasCoin)(ref Lattice!hasCoin lat, int endSite, bool isR) {
    int chainId = isR ? lat.sites[endSite].rChain : lat.sites[endSite].lChain;
    int newChainIdx = lat.chains[chainId].rootIdx - 1;
    Vec3 p = chainCentroid(&lat.chains[chainId].origin, newChainIdx);

    int nb = lat.allocSite(p);
    if (nb < 0) return -1;
    lat.chainPrepend(chainId, nb);
    lat.makeCrossChain(nb, !isR);
    return nb;
}

// ===========================================================================
//  Legacy wrapper — calls pureShift + handleOverflow + inline pruning
// ===========================================================================

ShiftResult applyShift(bool hasCoin)(ref Lattice!hasCoin lat, bool isR,
                                     double thresh2,
                                     double pruneThresh2 = 0,
                                     CoarseGrid* cgrid = null) {
    auto shift = pureShift!hasCoin(lat, isR);
    auto result = handleOverflow!hasCoin(lat, isR, shift.overflows, thresh2, cgrid);

    // Inline pruning (same as before)
    if (pruneThresh2 > 0) {
        foreach (ci; shift.activeChains) {
            auto ch = &lat.chains[ci];
            if (ch.ops.length <= 1) continue;

            // Forward end
            {
                int s = ch.ops[ch.ops.length - 1].siteId;
                double amp2 = spinorNorm2!hasCoin(lat, s);
                if (amp2 < pruneThresh2) {
                    result.probPruned += amp2;
                    if (cgrid !is null)
                        cgrid.addAmplitude(lat.sites[s].pos,
                                           &lat.psiRe[4*s], &lat.psiIm[4*s],
                                           lat.exitDirForSite(s, isR));
                    lat.psiRe[4*s .. 4*s+4] = 0;
                    lat.psiIm[4*s .. 4*s+4] = 0;
                    unlinkChainEnd!hasCoin(lat, s, isR, true);
                    result.nPruned++;
                }
            }

            if (ch.ops.length <= 1) continue;

            // Backward end
            {
                int s = ch.ops[0].siteId;
                double amp2 = spinorNorm2!hasCoin(lat, s);
                if (amp2 < pruneThresh2) {
                    result.probPruned += amp2;
                    if (cgrid !is null)
                        cgrid.addAmplitude(lat.sites[s].pos,
                                           &lat.psiRe[4*s], &lat.psiIm[4*s],
                                           lat.exitDirForSite(s, isR));
                    lat.psiRe[4*s .. 4*s+4] = 0;
                    lat.psiIm[4*s .. 4*s+4] = 0;
                    unlinkChainEnd!hasCoin(lat, s, isR, false);
                    result.nPruned++;
                }
            }
        }
    }

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
        if (!lat.hasChain(s, isR)) continue;

        Mat4 tau = makeTau(lat.exitDirForSite(s, isR));
        Mat4 Pp = projPlus(tau);
        Mat4 Pm = projMinus(tau);

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
                double[4] vRe = 0, vIm = 0;
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

    if (!lat.hasChain(nb, isR)) return true;

    Mat4 tau = makeTau(lat.exitDirForSite(nb, isR));
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
        ch.rootIdx++;
        for (int i = 0; i < ch.ops.length; i++) {
            int id = ch.ops[i].siteId;
            if (isR) lat.sites[id].rIdx--;
            else     lat.sites[id].lIdx--;
        }
    }

    if (isR) { lat.sites[s].rChain = -1; lat.sites[s].rIdx = -1; }
    else     { lat.sites[s].lChain = -1; lat.sites[s].lIdx = -1; }

    if (lat.sites[s].rChain < 0 && lat.sites[s].lChain < 0)
        lat.removeSite(s);
}

struct PruneResult { int count; double probPruned = 0; }

PruneResult pruneChainEnds(bool hasCoin)(ref Lattice!hasCoin lat, double thresh2) {
    PruneResult total;
    int prunedThisPass;

    do {
        prunedThisPass = 0;
        foreach (s; 0 .. lat.nsites) {
            if (!lat.hasChain(s, true) && !lat.hasChain(s, false)) continue;

            if (lat.hasChain(s, true) && lat.chainNext(s, true) == -1) {
                if (checkPruneEligible!hasCoin(lat, s, true, true, thresh2)) {
                    total.probPruned += spinorNorm2!hasCoin(lat, s);
                    unlinkChainEnd!hasCoin(lat, s, true, true);
                    prunedThisPass++; continue;
                }
            }
            if (lat.hasChain(s, true) && lat.chainPrev(s, true) == -1) {
                if (checkPruneEligible!hasCoin(lat, s, true, false, thresh2)) {
                    total.probPruned += spinorNorm2!hasCoin(lat, s);
                    unlinkChainEnd!hasCoin(lat, s, true, false);
                    prunedThisPass++; continue;
                }
            }
            if (lat.hasChain(s, false) && lat.chainNext(s, false) == -1) {
                if (checkPruneEligible!hasCoin(lat, s, false, true, thresh2)) {
                    total.probPruned += spinorNorm2!hasCoin(lat, s);
                    unlinkChainEnd!hasCoin(lat, s, false, true);
                    prunedThisPass++; continue;
                }
            }
            if (lat.hasChain(s, false) && lat.chainPrev(s, false) == -1) {
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
    import lattice : Lattice, ProximityGrid, generateSites;

    auto lat = Lattice!false.create(100000);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = ProximityGrid.create(maxChainLen * stepLen + 5.0, 0.35);
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

    auto res = applyShift(lat, false, 1e-20);

    double postNorm = 0;
    foreach (i; 0 .. 4 * lat.nsites)
        postNorm += lat.psiRe[i]*lat.psiRe[i] + lat.psiIm[i]*lat.psiIm[i];
    double total = postNorm + res.probAbsorbed;
    assert(total > 0.95 && total < 1.05, "Norm + absorbed out of range");
}

unittest {
    import lattice : Lattice, ProximityGrid, generateSites;

    auto lat = Lattice!true.create(100000);
    lat.coinCt = cos(0.5);
    lat.coinSt = sin(0.5);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = ProximityGrid.create(maxChainLen * stepLen + 5.0, 0.35);
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
