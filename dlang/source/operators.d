/**
 * operators.d — Walk step operators: shift, coin, V-mixing, and pruning.
 *
 * These operate on the lattice's psiRe/psiIm/tmpRe/tmpIm buffers and site
 * chain structure. Uses split real/imaginary storage (no Complex!double).
 * The walk step is: W = V_R · S_R · C_R · V_L · S_L · C_L
 */
module operators;

import std.math : sqrt, fabs, cos, sin, exp;
import geometry : Vec3, dot, norm, helixStep, reorth;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, matVecSplit, alpha;
import lattice : Lattice, nextFace, prevFace, PAT_R, PAT_L;

// ---- Spinor helpers ----

/// Norm squared of a 4-spinor at site s.
double spinorNorm2(const Lattice lat, int s) {
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

/// Try to extend a chain forward from site s. Returns new site ID or -1.
/// New sites at chain ends are guaranteed unique (BC helix no-loops property).
private int tryExtendFwd(ref Lattice lat, int s, bool isR,
                         const int[4] pat, ref double[4] shiftedRe,
                         ref double[4] shiftedIm, double thresh2) {
    double amp2 = 0;
    foreach (a; 0 .. 4) amp2 += shiftedRe[a] * shiftedRe[a] + shiftedIm[a] * shiftedIm[a];
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

/// Try to extend a chain backward from site s. Returns new site ID or -1.
private int tryExtendBwd(ref Lattice lat, int s, bool isR,
                         const int[4] pat, ref double[4] shiftedRe,
                         ref double[4] shiftedIm, double thresh2) {
    double amp2 = 0;
    foreach (a; 0 .. 4) amp2 += shiftedRe[a] * shiftedRe[a] + shiftedIm[a] * shiftedIm[a];
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

/// Apply the shift operator for one chirality. Reads from psi, writes to tmp.
/// Swaps psi/tmp at the end.
ShiftResult applyShift(ref Lattice lat, bool isR, const int[4] pat, double thresh2) {
    int ns = lat.nsites;
    ShiftResult result;

    // Zero tmp for current sites
    lat.tmpRe[0 .. 4 * ns] = 0;
    lat.tmpIm[0 .. 4 * ns] = 0;

    foreach (s; 0 .. ns) {
        int face = lat.chainFace(s, isR);

        if (face < 0) {
            // Not on a chain -- identity
            lat.tmpRe[4*s .. 4*s+4] = lat.psiRe[4*s .. 4*s+4];
            lat.tmpIm[4*s .. 4*s+4] = lat.psiIm[4*s .. 4*s+4];
            continue;
        }

        Vec3 dv = lat.sites[s].dirs[face];
        Mat4 tau = makeTau(dv);
        Mat4 Pp = projPlus(tau);
        Mat4 Pm = projMinus(tau);

        // P+ component -> forward
        {
            double[4] shiftedRe = 0, shiftedIm = 0;
            matVecSplit(Pp, &lat.psiRe[4*s], &lat.psiIm[4*s],
                        shiftedRe.ptr, shiftedIm.ptr);

            int nb = lat.chainNext(s, isR);
            if (nb < 0)
                nb = tryExtendFwd(lat, s, isR, pat, shiftedRe, shiftedIm, thresh2);

            if (nb >= 0) {
                int fn = lat.chainFace(nb, isR);
                Vec3 dn = lat.sites[nb].dirs[fn];
                Mat4 tn = makeTau(dn);
                Mat4 U = frameTransport(tau, tn);
                Mat4 bl = mul(U, Pp);
                double[4] resRe = 0, resIm = 0;
                matVecSplit(bl, &lat.psiRe[4*s], &lat.psiIm[4*s],
                            resRe.ptr, resIm.ptr);
                foreach (a; 0 .. 4) {
                    lat.tmpRe[4*nb + a] += resRe[a];
                    lat.tmpIm[4*nb + a] += resIm[a];
                }
                if (nb >= ns) result.nCreated++;
            } else {
                foreach (a; 0 .. 4)
                    result.probAbsorbed += shiftedRe[a] * shiftedRe[a] + shiftedIm[a] * shiftedIm[a];
            }
        }

        // P- component -> backward
        {
            double[4] shiftedRe = 0, shiftedIm = 0;
            matVecSplit(Pm, &lat.psiRe[4*s], &lat.psiIm[4*s],
                        shiftedRe.ptr, shiftedIm.ptr);

            int nb = lat.chainPrev(s, isR);
            if (nb < 0)
                nb = tryExtendBwd(lat, s, isR, pat, shiftedRe, shiftedIm, thresh2);

            if (nb >= 0) {
                int fp = lat.chainFace(nb, isR);
                Vec3 dp = lat.sites[nb].dirs[fp];
                Mat4 tp = makeTau(dp);
                Mat4 U = frameTransport(tau, tp);
                Mat4 bl = mul(U, Pm);
                double[4] resRe = 0, resIm = 0;
                matVecSplit(bl, &lat.psiRe[4*s], &lat.psiIm[4*s],
                            resRe.ptr, resIm.ptr);
                foreach (a; 0 .. 4) {
                    lat.tmpRe[4*nb + a] += resRe[a];
                    lat.tmpIm[4*nb + a] += resIm[a];
                }
                if (nb >= ns) result.nCreated++;
            } else {
                foreach (a; 0 .. 4)
                    result.probAbsorbed += shiftedRe[a] * shiftedRe[a] + shiftedIm[a] * shiftedIm[a];
            }
        }
    }

    lat.swapBuffers();
    return result;
}

// ---- Coin operator ----

/// Apply exp(-i theta (d . alpha)) to spinor at site n, in place.
private void applyCoinDir(ref Lattice lat, double[3] d, int n, double ct, double st) {
    // Build coin matrix: C_{ab} = cos(theta) delta_{ab} - i sin(theta) (d . alpha)_{ab}
    Mat4 coinMat;
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
            // (a - ib)(c + id) = ac + bd + i(ad - bc) where a=0,b=st,c=eaRe,d=eaIm
            // -i*st * (eaRe + i*eaIm) = st*eaIm - i*st*eaRe
            coinMat.re[4*a+b] = (a == b ? ct : 0.0) + st * eaIm;
            coinMat.im[4*a+b] = -st * eaRe;
        }

    double[4] resRe = 0, resIm = 0;
    matVecSplit(coinMat, &lat.psiRe[4*n], &lat.psiIm[4*n], resRe.ptr, resIm.ptr);
    lat.psiRe[4*n .. 4*n+4] = resRe[];
    lat.psiIm[4*n .. 4*n+4] = resIm[];
}

/// Apply coin operator for one chirality, in place.
/// coinType: 0 = e . alpha, 1 = dual parity (f1 . alpha then f2 . alpha, both perp e).
void applyCoin(ref Lattice lat, bool isR, double ct, double st, int coinType) {
    int ns = lat.nsites;
    foreach (n; 0 .. ns) {
        int face = lat.chainFace(n, isR);
        if (face < 0) continue;

        Vec3 e = lat.sites[n].dirs[face];
        double en = norm(e);
        if (en < 1e-15) continue;
        Vec3 ehat = e * (1.0 / en);

        if (coinType == 0) {
            applyCoinDir(lat, [e.x, e.y, e.z], n, ct, st);
        } else {
            // Dual parity: two perpendicular directions
            Vec3 f1 = Vec3(ehat.y, -ehat.x, 0);
            double fn = norm(f1);
            if (fn < 1e-10) {
                f1 = Vec3(-ehat.z, 0, ehat.x);
                fn = norm(f1);
            }
            f1 = f1 * (1.0 / fn);
            // f2 = ehat x f1
            Vec3 f2 = Vec3(
                ehat.y * f1.z - ehat.z * f1.y,
                ehat.z * f1.x - ehat.x * f1.z,
                ehat.x * f1.y - ehat.y * f1.x,
            );
            fn = norm(f2);
            f2 = f2 * (1.0 / fn);

            applyCoinDir(lat, [f1.x, f1.y, f1.z], n, ct, st);
            applyCoinDir(lat, [f2.x, f2.y, f2.z], n, ct, st);
        }
    }
}

// ---- V mixing ----

/// Apply post-shift V mixing: V = cos(phi)I + i sin(phi) M
/// where M swaps P+ and P- eigenspaces of the local tau.
void applyVmix(ref Lattice lat, bool isR, double mixPhi) {
    if (mixPhi == 0.0) return;
    double cp = cos(mixPhi), sp = sin(mixPhi);
    int ns = lat.nsites;

    foreach (s; 0 .. ns) {
        int face = lat.chainFace(s, isR);
        if (face < 0) continue;

        Mat4 tau = makeTau(lat.sites[s].dirs[face]);
        Mat4 Pp = projPlus(tau);
        Mat4 Pm = projMinus(tau);

        // Find orthonormal bases for P+ and P- via Gram-Schmidt
        // ppBasis[a][j] and pmBasis[a][j]: component a of basis vector j
        double[4][2] ppBRe = 0, ppBIm = 0, pmBRe = 0, pmBIm = 0;
        int npFound = 0, nmFound = 0;

        foreach (col; 0 .. 4) {
            if (npFound >= 2 && nmFound >= 2) break;

            if (npFound < 2) {
                double[4] vRe = void, vIm = void;
                foreach (a; 0 .. 4) { vRe[a] = Pp.re[4*a+col]; vIm[a] = Pp.im[4*a+col]; }
                // Orthogonalize against existing basis vectors
                foreach (j; 0 .. npFound) {
                    // d = <basis_j | v> = conj(basis_j) . v
                    double dRe = 0, dIm = 0;
                    foreach (a; 0 .. 4) {
                        dRe += ppBRe[a][j] * vRe[a] + ppBIm[a][j] * vIm[a];
                        dIm += ppBRe[a][j] * vIm[a] - ppBIm[a][j] * vRe[a];
                    }
                    // v -= d * basis_j
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
                double[4] vRe = void, vIm = void;
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
        // V = cos(phi)I + i sin(phi) M
        Mat4 V;
        foreach (j; 0 .. 2)
            foreach (a; 0 .. 4)
                foreach (b; 0 .. 4) {
                    int ab = 4*a+b;
                    // |pm_j><pp_j|: outer product pm_j * conj(pp_j)
                    // (pmRe + i pmIm)(ppRe - i ppIm) = pmRe*ppRe + pmIm*ppIm + i(pmIm*ppRe - pmRe*ppIm)
                    V.re[ab] += pmBRe[a][j] * ppBRe[b][j] + pmBIm[a][j] * ppBIm[b][j];
                    V.im[ab] += pmBIm[a][j] * ppBRe[b][j] - pmBRe[a][j] * ppBIm[b][j];
                    // |pp_j><pm_j|
                    V.re[ab] += ppBRe[a][j] * pmBRe[b][j] + ppBIm[a][j] * pmBIm[b][j];
                    V.im[ab] += ppBIm[a][j] * pmBRe[b][j] - ppBRe[a][j] * pmBIm[b][j];
                }
        // V = cos(phi)I + i sin(phi) M
        // i * (Mre + i Mim) = -Mim + i Mre
        foreach (a; 0 .. 4)
            foreach (b; 0 .. 4) {
                int ab = 4*a+b;
                double mRe = V.re[ab], mIm = V.im[ab];
                V.re[ab] = (a == b ? cp : 0.0) + sp * (-mIm);
                V.im[ab] = sp * mRe;
            }

        // Apply V to psi at site s
        double[4] resRe = 0, resIm = 0;
        matVecSplit(V, &lat.psiRe[4*s], &lat.psiIm[4*s], resRe.ptr, resIm.ptr);
        lat.psiRe[4*s .. 4*s+4] = resRe[];
        lat.psiIm[4*s .. 4*s+4] = resIm[];
    }
}

// ---- Chain-end pruning ----

/// Check if a chain-end site is eligible for pruning.
private bool checkPruneEligible(const Lattice lat, int s, bool isR, bool isFwd, double thresh2) {
    double amp2 = spinorNorm2(lat, s);
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
        flow2 += vRe[a] * vRe[a] + vIm[a] * vIm[a];
    return flow2 < thresh2;
}

/// Unlink a chain-end site by truncating the chain.
/// If orphaned from both chains, remove the site entirely.
private void unlinkChainEnd(ref Lattice lat, int s, bool isR, bool isFwd) {
    int chainId = isR ? lat.sites[s].rChain : lat.sites[s].lChain;
    if (chainId < 0) return;

    auto ch = &lat.chains[chainId];
    if (isFwd) {
        // Remove from forward end (last element)
        assert(ch.siteIds[$ - 1] == s);
        ch.siteIds = ch.siteIds[0 .. $ - 1];
    } else {
        // Remove from backward end (first element)
        assert(ch.siteIds[0] == s);
        ch.siteIds = ch.siteIds[1 .. $];
        ch.rootIdx--;
        // Shift indices of remaining sites
        foreach (id; ch.siteIds) {
            if (isR) lat.sites[id].rIdx--;
            else     lat.sites[id].lIdx--;
        }
    }

    // Clear chain membership for this site
    if (isR) { lat.sites[s].rChain = -1; lat.sites[s].rIdx = -1; lat.sites[s].rFace = -1; }
    else     { lat.sites[s].lChain = -1; lat.sites[s].lIdx = -1; lat.sites[s].lFace = -1; }

    // If orphaned from both chains, remove entirely
    if (lat.sites[s].rChain < 0 && lat.sites[s].lChain < 0)
        lat.removeSite(s);
}

/// Prune low-amplitude chain-end sites. Iterates until stable.
/// Returns (count pruned, probability removed).
struct PruneResult { int count; double probPruned; }

PruneResult pruneChainEnds(ref Lattice lat, double thresh2) {
    PruneResult total;
    int prunedThisPass;

    do {
        prunedThisPass = 0;
        foreach (s; 0 .. lat.nsites) {
            if (lat.chainFace(s, true) < 0 && lat.chainFace(s, false) < 0) continue;

            // R-chain forward end
            if (lat.chainFace(s, true) >= 0 && lat.chainNext(s, true) == -1) {
                if (checkPruneEligible(lat, s, true, true, thresh2)) {
                    total.probPruned += spinorNorm2(lat, s);
                    unlinkChainEnd(lat, s, true, true);
                    prunedThisPass++;
                    continue;
                }
            }
            // R-chain backward end
            if (lat.chainFace(s, true) >= 0 && lat.chainPrev(s, true) == -1) {
                if (checkPruneEligible(lat, s, true, false, thresh2)) {
                    total.probPruned += spinorNorm2(lat, s);
                    unlinkChainEnd(lat, s, true, false);
                    prunedThisPass++;
                    continue;
                }
            }
            // L-chain forward end
            if (lat.chainFace(s, false) >= 0 && lat.chainNext(s, false) == -1) {
                if (checkPruneEligible(lat, s, false, true, thresh2)) {
                    total.probPruned += spinorNorm2(lat, s);
                    unlinkChainEnd(lat, s, false, true);
                    prunedThisPass++;
                    continue;
                }
            }
            // L-chain backward end
            if (lat.chainFace(s, false) >= 0 && lat.chainPrev(s, false) == -1) {
                if (checkPruneEligible(lat, s, false, false, thresh2)) {
                    total.probPruned += spinorNorm2(lat, s);
                    unlinkChainEnd(lat, s, false, false);
                    prunedThisPass++;
                    continue;
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

    // Generate a small lattice and run one shift step
    auto lat = Lattice.create(100000);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    // Initialize Gaussian wavepacket in first spinor component
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

    // Check initial norm = 1
    double initNorm = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        initNorm += lat.psiRe[i] * lat.psiRe[i] + lat.psiIm[i] * lat.psiIm[i];
    }
    assert(fabs(initNorm - 1.0) < 1e-12);

    // Apply one L-shift
    auto res = applyShift(lat, false, PAT_L, 1e-20);

    // Norm should be approximately preserved (some absorption at boundary)
    double postNorm = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        postNorm += lat.psiRe[i] * lat.psiRe[i] + lat.psiIm[i] * lat.psiIm[i];
    }
    // Norm + absorbed should be close to 1. Small excess is from new sites
    // created during the shift that receive amplitude from boundary sites.
    double total = postNorm + res.probAbsorbed;
    assert(total > 0.95 && total < 1.05, "Norm + absorbed out of range");
}

unittest {
    import lattice : Lattice, DensityGrid, generateSites;

    // Coin preserves norm (theta=0 is identity)
    auto lat = Lattice.create(100000);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    // Set psi to something nonzero
    double val = 1.0 / sqrt(cast(double) lat.nsites);
    foreach (n; 0 .. lat.nsites)
        lat.psiRe[4*n] = val;

    double normBefore = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        normBefore += lat.psiRe[i] * lat.psiRe[i] + lat.psiIm[i] * lat.psiIm[i];
    }

    // Apply coin with theta=0.5
    applyCoin(lat, true, cos(0.5), sin(0.5), 1);

    double normAfter = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        normAfter += lat.psiRe[i] * lat.psiRe[i] + lat.psiIm[i] * lat.psiIm[i];
    }

    assert(fabs(normAfter - normBefore) < 1e-10, "Coin should preserve norm");
}
