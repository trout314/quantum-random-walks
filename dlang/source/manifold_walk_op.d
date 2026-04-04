/**
 * manifold_walk_op.d — Walk operator for a triangulated manifold.
 *
 * Applies the quantum walk (shift + V-mixing) to a wavefunction stored
 * on the sites of a TriangulationWalk. Chains are open in the site
 * container; the operator handles chain-end overflow via the redirect
 * table to close the walk on the manifold.
 *
 * Shift blocks (frame transport × projection) are computed on-the-fly
 * from the geometric data in the SiteContainer, not precomputed.
 */
module manifold_walk_op;

import site_container : SiteContainer, GeoChain;
import triangulation_walk : TriangulationWalk, Redirect;
import geometry : Vec3, chainExitDir;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul;

/// Walk state: psi arrays + reference to the geometry/topology.
struct ManifoldWalkState {
    double[] psiRe, psiIm;
    double[] tmpRe, tmpIm;
    int nsites;

    static ManifoldWalkState create(int n) {
        ManifoldWalkState ws;
        ws.nsites = n;
        ws.psiRe = new double[4 * n]; ws.psiRe[] = 0;
        ws.psiIm = new double[4 * n]; ws.psiIm[] = 0;
        ws.tmpRe = new double[4 * n]; ws.tmpRe[] = 0;
        ws.tmpIm = new double[4 * n]; ws.tmpIm[] = 0;
        return ws;
    }

    void swapBuffers() {
        auto sr = psiRe; psiRe = tmpRe; tmpRe = sr;
        auto si = psiIm; psiIm = tmpIm; tmpIm = si;
    }

    double norm2() const {
        double n2 = 0;
        foreach (i; 0 .. 4 * nsites)
            n2 += psiRe[i] * psiRe[i] + psiIm[i] * psiIm[i];
        return n2;
    }

    void zero() {
        psiRe[0 .. 4 * nsites] = 0;
        psiIm[0 .. 4 * nsites] = 0;
    }
}

/// Multiply 4×4 complex matrix by complex 4-vector (split real/imag).
private void matVec4(ref const Mat4 M,
                     const(double)* inRe, const(double)* inIm,
                     double* outRe, double* outIm)
{
    foreach (a; 0 .. 4) {
        double re = 0, im = 0;
        foreach (b; 0 .. 4) {
            int ab = 4 * a + b;
            re += M.re[ab] * inRe[b] - M.im[ab] * inIm[b];
            im += M.re[ab] * inIm[b] + M.im[ab] * inRe[b];
        }
        outRe[a] = re;
        outIm[a] = im;
    }
}

/// Get the exit direction for a site on a chain, using the chain's analytic formula.
private Vec3 siteExitDir(ref const SiteContainer sc, int siteId, bool isR) {
    return sc.exitDirForSite(siteId, isR);
}

/// Compute the forward shift block: frameTransport(tau_here, tau_next) × P+(tau_here)
private Mat4 fwdBlock(Vec3 exitHere, Vec3 exitNext) {
    Mat4 tau = makeTau(exitHere);
    Mat4 tauN = makeTau(exitNext);
    return mul(frameTransport(tau, tauN), projPlus(tau));
}

/// Compute the backward shift block: frameTransport(tau_here, tau_prev) × P-(tau_here)
private Mat4 bwdBlock(Vec3 exitHere, Vec3 exitPrev) {
    Mat4 tau = makeTau(exitHere);
    Mat4 tauP = makeTau(exitPrev);
    return mul(frameTransport(tau, tauP), projMinus(tau));
}

/// Apply the shift operator for one chirality (all chains of that type).
/// Chain-end overflow is returned for redirect handling.
///
/// This is the 1D shift applied independently to each chain.
struct ChainOverflow {
    int chainId;
    bool isFwd;
    double[4] ampRe = 0;
    double[4] ampIm = 0;
    Vec3 exitDir;
    int endSiteId;
}

ChainOverflow[] applyShift(ref const SiteContainer sc, ref ManifoldWalkState ws, bool isR) {
    int ns = ws.nsites;

    // Zero tmp
    ws.tmpRe[0 .. 4 * ns] = 0;
    ws.tmpIm[0 .. 4 * ns] = 0;

    // Copy identity for sites not on a chain of this type
    foreach (s; 0 .. ns) {
        if (!sc.hasChain(s, isR)) {
            ws.tmpRe[4*s .. 4*s+4] = ws.psiRe[4*s .. 4*s+4];
            ws.tmpIm[4*s .. 4*s+4] = ws.psiIm[4*s .. 4*s+4];
        }
    }

    ChainOverflow[] overflows;

    foreach (ci, ref ch; sc.chains) {
        if (ch.isR != isR) continue;
        int n = ch.length;
        if (n == 0) continue;

        foreach (i; 0 .. n) {
            int s = ch.siteAt(i);
            Vec3 exitS = siteExitDir(sc, s, isR);
            double[4] accRe = 0, accIm = 0;

            // Gather from predecessor
            int prevIdx = i - 1;
            if (prevIdx >= 0) {
                int prev = ch.siteAt(prevIdx);
                Vec3 exitP = siteExitDir(sc, prev, isR);
                Mat4 fb = fwdBlock(exitP, exitS);
                double[4] tRe = 0, tIm = 0;
                matVec4(fb, &ws.psiRe[4*prev], &ws.psiIm[4*prev],
                        tRe.ptr, tIm.ptr);
                foreach (a; 0 .. 4) { accRe[a] += tRe[a]; accIm[a] += tIm[a]; }
            }

            // Gather from successor
            int nextIdx = i + 1;
            if (nextIdx < n) {
                int next = ch.siteAt(nextIdx);
                Vec3 exitN = siteExitDir(sc, next, isR);
                Mat4 bb = bwdBlock(exitN, exitS);
                double[4] tRe = 0, tIm = 0;
                matVec4(bb, &ws.psiRe[4*next], &ws.psiIm[4*next],
                        tRe.ptr, tIm.ptr);
                foreach (a; 0 .. 4) { accRe[a] += tRe[a]; accIm[a] += tIm[a]; }
            }

            foreach (a; 0 .. 4) {
                ws.tmpRe[4*s + a] += accRe[a];
                ws.tmpIm[4*s + a] += accIm[a];
            }
        }

        // Compute overflow at chain ends
        // Forward end: P+ at last site
        {
            int lastSite = ch.siteAt(n - 1);
            Vec3 exitLast = siteExitDir(sc, lastSite, isR);
            Mat4 tau = makeTau(exitLast);
            Mat4 Pp = projPlus(tau);
            ChainOverflow ov;
            ov.chainId = cast(int) ci;
            ov.isFwd = true;
            ov.exitDir = exitLast;
            ov.endSiteId = lastSite;
            matVec4(Pp, &ws.psiRe[4*lastSite], &ws.psiIm[4*lastSite],
                    ov.ampRe.ptr, ov.ampIm.ptr);
            overflows ~= ov;
        }
        // Backward end: P- at first site
        {
            int firstSite = ch.siteAt(0);
            Vec3 exitFirst = siteExitDir(sc, firstSite, isR);
            Mat4 tau = makeTau(exitFirst);
            Mat4 Pm = projMinus(tau);
            ChainOverflow ov;
            ov.chainId = cast(int) ci;
            ov.isFwd = false;
            ov.exitDir = exitFirst;
            ov.endSiteId = firstSite;
            matVec4(Pm, &ws.psiRe[4*firstSite], &ws.psiIm[4*firstSite],
                    ov.ampRe.ptr, ov.ampIm.ptr);
            overflows ~= ov;
        }
    }

    ws.swapBuffers();
    return overflows;
}

/// Apply redirects: frame-transport overflow amplitude to target sites.
void applyRedirects(ref const SiteContainer sc,
                    ref ManifoldWalkState ws,
                    const(ChainOverflow)[] overflows,
                    const(Redirect)[] redirects)
{
    foreach (ref rd; redirects) {
        // Find matching overflow
        foreach (ref ov; overflows) {
            if (ov.chainId == rd.chainId && ov.isFwd == rd.isFwd) {
                // Frame-transport to target
                Vec3 targetExit = sc.exitDirForSite(rd.targetSite,
                    sc.chains[rd.chainId].isR);
                Mat4 tauEnd = makeTau(ov.exitDir);
                Mat4 tauTarget = makeTau(targetExit);
                Mat4 U = frameTransport(tauEnd, tauTarget);

                double[4] resRe = 0, resIm = 0;
                matVec4(U, ov.ampRe.ptr, ov.ampIm.ptr, resRe.ptr, resIm.ptr);

                foreach (a; 0 .. 4) {
                    ws.psiRe[4 * rd.targetSite + a] += resRe[a];
                    ws.psiIm[4 * rd.targetSite + a] += resIm[a];
                }
                break;
            }
        }
    }
}

/// Apply V-mixing (coin) for one chirality.
/// Uses complex Gram-Schmidt to extract P± eigenspace bases,
/// then applies V = cos(φ)I + i sin(φ) M where M swaps eigenspaces.
void applyVmix(ref const SiteContainer sc, ref ManifoldWalkState ws,
               bool isR, double mixPhi)
{
    import std.math : cos, sin, sqrt;
    if (mixPhi == 0) return;

    double cp = cos(mixPhi), sp = sin(mixPhi);
    int ns = ws.nsites;

    foreach (s; 0 .. ns) {
        if (!sc.hasChain(s, isR)) continue;
        Vec3 exitD = sc.exitDirForSite(s, isR);
        Mat4 tau = makeTau(exitD);
        Mat4 Pp = projPlus(tau);
        Mat4 Pm = projMinus(tau);

        // Complex Gram-Schmidt for P+ and P- eigenspace bases
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

        // Build V = cos(φ)I + i sin(φ) M as a 4×4 matrix, then apply.
        // M = Σ_j (|pm_j⟩⟨pp_j| + |pp_j⟩⟨pm_j|)
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
        // V_final = cos(φ) I + i sin(φ) M → re part: cos(φ) δ_ab - sin(φ) M.im
        //                                  → im part: sin(φ) M.re
        foreach (a; 0 .. 4)
            foreach (b; 0 .. 4) {
                int ab = 4*a+b;
                double mRe = V.re[ab], mIm = V.im[ab];
                V.re[ab] = (a == b ? cp : 0.0) + sp * (-mIm);
                V.im[ab] = sp * mRe;
            }

        // Apply V to psi at this site
        double[4] outRe = 0, outIm = 0;
        matVec4(V, &ws.psiRe[4*s], &ws.psiIm[4*s], outRe.ptr, outIm.ptr);
        ws.psiRe[4*s .. 4*s+4] = outRe[];
        ws.psiIm[4*s .. 4*s+4] = outIm[];
    }
}

/// One full walk step: S_L → V_L → S_R → V_R with redirect handling.
double manifoldStep(ref const TriangulationWalk tw, ref ManifoldWalkState ws, double mixPhi) {
    // L-shift
    auto ovL = applyShift(tw.sites, ws, false);
    applyRedirects(tw.sites, ws, ovL, tw.redirects);
    applyVmix(tw.sites, ws, false, mixPhi);

    // R-shift
    auto ovR = applyShift(tw.sites, ws, true);
    applyRedirects(tw.sites, ws, ovR, tw.redirects);
    applyVmix(tw.sites, ws, true, mixPhi);

    return ws.norm2();
}
