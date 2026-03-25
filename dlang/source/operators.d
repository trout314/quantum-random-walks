/**
 * operators.d — Walk step operators: shift, coin, V-mixing, and pruning.
 *
 * These operate on the lattice's psi/tmp buffers and site chain structure.
 * The walk step is: W = V_R · S_R · C_R · V_L · S_L · C_L
 */
module operators;

import std.math : sqrt, fabs, cos, sin, exp;
import std.complex : Complex;
import geometry : Vec3, dot, norm, helixStep, reorth;
import dirac : Mat4, C, makeTau, projPlus, projMinus, frameTransport, mul, conj, alpha;
import lattice : Lattice, nextFace, prevFace, PAT_R, PAT_L;

// ---- Spinor helpers ----

/// Norm squared of a 4-spinor at site s.
double spinorNorm2(const Lattice lat, int s) {
    double n = 0;
    foreach (a; 0 .. 4) {
        auto z = lat.psi[4*s + a];
        n += z.re * z.re + z.im * z.im;
    }
    return n;
}

/// Apply a Mat4 to psi at site s, write result to out[0..4].
void matVecPsi(Mat4 M, const C[] psi, int s, ref C[4] result) {
    foreach (a; 0 .. 4) {
        C acc = C(0, 0);
        foreach (b; 0 .. 4)
            acc = acc + M[a, b] * psi[4*s + b];
        result[a] = acc;
    }
}

// ---- Adaptive shift operator ----

struct ShiftResult {
    int nCreated = 0;
    double probAbsorbed = 0;
}

/// Try to extend a chain forward from site s. Returns new site ID or -1.
private int tryExtendFwd(ref Lattice lat, int s, bool isR,
                         const int[4] pat, ref C[4] shifted, double thresh2) {
    double amp2 = 0;
    foreach (a; 0 .. 4) amp2 += shifted[a].re * shifted[a].re + shifted[a].im * shifted[a].im;
    if (amp2 < thresh2) return -1;

    int face = lat.chainFace(s, isR);
    int nf = nextFace(pat, face);

    Vec3 p = lat.sites[s].pos;
    auto d = lat.sites[s].dirs;
    helixStep(p, d, face);

    int nb = lat.findSite(p);
    if (nb >= 0) {
        // Existing site from another chain — link but absorb
        lat.setChainNext(s, isR, nb);
        if (lat.chainPrev(nb, isR) < 0) {
            lat.setChainPrev(nb, isR, s);
            lat.setChainFace(nb, isR, nf);
        }
        return -1;
    }

    reorth(d);
    nb = lat.insertSite(p, d);
    lat.linkForward(s, nb, isR, nf);
    return nb;
}

/// Try to extend a chain backward from site s. Returns new site ID or -1.
private int tryExtendBwd(ref Lattice lat, int s, bool isR,
                         const int[4] pat, ref C[4] shifted, double thresh2) {
    double amp2 = 0;
    foreach (a; 0 .. 4) amp2 += shifted[a].re * shifted[a].re + shifted[a].im * shifted[a].im;
    if (amp2 < thresh2) return -1;

    int face = lat.chainFace(s, isR);
    int pf = prevFace(pat, face);

    Vec3 p = lat.sites[s].pos;
    auto d = lat.sites[s].dirs;
    helixStep(p, d, pf);

    int nb = lat.findSite(p);
    if (nb >= 0) {
        lat.setChainPrev(s, isR, nb);
        if (lat.chainNext(nb, isR) < 0) {
            lat.setChainNext(nb, isR, s);
            lat.setChainFace(nb, isR, pf);
        }
        return -1;
    }

    reorth(d);
    nb = lat.insertSite(p, d);
    lat.linkBackward(s, nb, isR, pf);
    return nb;
}

/// Apply the shift operator for one chirality. Reads from psi, writes to tmp.
/// Swaps psi/tmp at the end.
ShiftResult applyShift(ref Lattice lat, bool isR, const int[4] pat, double thresh2) {
    int ns = lat.nsites;
    ShiftResult result;

    // Zero tmp for current sites
    lat.tmp[0 .. 4 * ns] = C(0, 0);

    foreach (s; 0 .. ns) {
        int face = lat.chainFace(s, isR);

        if (face < 0) {
            // Not on a chain — identity
            lat.tmp[4*s .. 4*s+4] = lat.psi[4*s .. 4*s+4];
            continue;
        }

        Vec3 dv = lat.sites[s].dirs[face];
        Mat4 tau = makeTau(dv);
        Mat4 Pp = projPlus(tau);
        Mat4 Pm = projMinus(tau);

        // P+ component -> forward
        {
            C[4] shifted = [C(0,0), C(0,0), C(0,0), C(0,0)];
            matVecPsi(Pp, lat.psi, s, shifted);

            int nb = lat.chainNext(s, isR);
            // Clear stale link to pruned site
            if (nb >= 0 && lat.chainFace(nb, isR) < 0) {
                lat.setChainNext(s, isR, -1);
                nb = -1;
            }
            if (nb < 0)
                nb = tryExtendFwd(lat, s, isR, pat, shifted, thresh2);

            if (nb >= 0) {
                int fn = lat.chainFace(nb, isR);
                Vec3 dn = lat.sites[nb].dirs[fn];
                Mat4 tn = makeTau(dn);
                Mat4 U = frameTransport(tau, tn);
                Mat4 bl = mul(U, Pp);
                C[4] res;
                matVecPsi(bl, lat.psi, s, res);
                foreach (a; 0 .. 4)
                    lat.tmp[4*nb + a] = lat.tmp[4*nb + a] + res[a];
                if (nb >= ns) result.nCreated++;
            } else {
                foreach (a; 0 .. 4) {
                    auto z = shifted[a];
                    result.probAbsorbed += z.re * z.re + z.im * z.im;
                }
            }
        }

        // P- component -> backward
        {
            C[4] shifted = [C(0,0), C(0,0), C(0,0), C(0,0)];
            matVecPsi(Pm, lat.psi, s, shifted);

            int nb = lat.chainPrev(s, isR);
            if (nb >= 0 && lat.chainFace(nb, isR) < 0) {
                lat.setChainPrev(s, isR, -1);
                nb = -1;
            }
            if (nb < 0)
                nb = tryExtendBwd(lat, s, isR, pat, shifted, thresh2);

            if (nb >= 0) {
                int fp = lat.chainFace(nb, isR);
                Vec3 dp = lat.sites[nb].dirs[fp];
                Mat4 tp = makeTau(dp);
                Mat4 U = frameTransport(tau, tp);
                Mat4 bl = mul(U, Pm);
                C[4] res;
                matVecPsi(bl, lat.psi, s, res);
                foreach (a; 0 .. 4)
                    lat.tmp[4*nb + a] = lat.tmp[4*nb + a] + res[a];
                if (nb >= ns) result.nCreated++;
            } else {
                foreach (a; 0 .. 4) {
                    auto z = shifted[a];
                    result.probAbsorbed += z.re * z.re + z.im * z.im;
                }
            }
        }
    }

    lat.swapBuffers();
    return result;
}

// ---- Coin operator ----

/// Apply exp(-i θ (d·α)) to spinor at site n, in place.
private void applyCoinDir(ref Lattice lat, double[3] d, int n, double ct, double st) {
    C[4] result;
    foreach (a; 0 .. 4) {
        C acc = C(0, 0);
        foreach (b; 0 .. 4) {
            // (d · α)_{ab}
            C ea = C(0, 0);
            foreach (k; 0 .. 3)
                ea = ea + C(d[k], 0) * alpha(k)[a, b];
            // C_{ab} = cos(θ) δ_{ab} - i sin(θ) (d·α)_{ab}
            C cab = C(ct * (a == b ? 1.0 : 0.0), 0) - C(0, st) * ea;
            acc = acc + cab * lat.psi[4*n + b];
        }
        result[a] = acc;
    }
    lat.psi[4*n .. 4*n+4] = result[];
}

/// Apply coin operator for one chirality, in place.
/// coinType: 0 = e·α, 1 = dual parity (f₁·α then f₂·α, both ⊥ e).
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
            // f2 = ehat × f1
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

/// Apply post-shift V mixing: V = cos(φ)I + i sin(φ) M
/// where M swaps P+ and P- eigenspaces of the local τ.
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
        C[4][2] ppBasis, pmBasis;
        int npFound = 0, nmFound = 0;

        foreach (col; 0 .. 4) {
            if (npFound >= 2 && nmFound >= 2) break;

            if (npFound < 2) {
                C[4] v;
                foreach (a; 0 .. 4) v[a] = Pp[a, col];
                foreach (j; 0 .. npFound) {
                    C d = C(0, 0);
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
                    C d = C(0, 0);
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

        // M = Σ_j |pm_j><pp_j| + |pp_j><pm_j|
        // V = cos(φ)I + i sin(φ) M
        Mat4 V = Mat4(0);
        foreach (j; 0 .. 2)
            foreach (a; 0 .. 4)
                foreach (b; 0 .. 4) {
                    V[a, b] = V[a, b] + pmBasis[a][j] * conj(ppBasis[b][j]);
                    V[a, b] = V[a, b] + ppBasis[a][j] * conj(pmBasis[b][j]);
                }
        foreach (a; 0 .. 4)
            foreach (b; 0 .. 4)
                V[a, b] = C(cp * (a == b ? 1.0 : 0.0), 0) + C(0, sp) * V[a, b];

        // Apply V to psi at site s
        C[4] result;
        foreach (a; 0 .. 4) {
            C acc = C(0, 0);
            foreach (b; 0 .. 4)
                acc = acc + V[a, b] * lat.psi[4*s + b];
            result[a] = acc;
        }
        lat.psi[4*s .. 4*s+4] = result[];
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

    double flow2 = 0;
    foreach (a; 0 .. 4) {
        C v = C(0, 0);
        foreach (b; 0 .. 4)
            v = v + P[a, b] * lat.psi[4*nb + b];
        flow2 += v.re * v.re + v.im * v.im;
    }
    return flow2 < thresh2;
}

/// Unlink a chain-end site. If orphaned, remove entirely.
private void unlinkChainEnd(ref Lattice lat, int s, bool isR, bool isFwd) {
    if (isFwd) {
        int nb = lat.chainPrev(s, isR);
        if (nb >= 0) lat.setChainNext(nb, isR, -1);
        lat.setChainPrev(s, isR, -1);
        lat.setChainFace(s, isR, -1);
    } else {
        int nb = lat.chainNext(s, isR);
        if (nb >= 0) lat.setChainPrev(nb, isR, -1);
        lat.setChainNext(s, isR, -1);
        lat.setChainFace(s, isR, -1);
    }

    if (lat.chainFace(s, true) < 0 && lat.chainFace(s, false) < 0)
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
    auto lat = Lattice.create(100000, 17);
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
        lat.psi[4*n] = C(w, 0);
        norm2 += w * w;
    }
    double normFactor = 1.0 / sqrt(norm2);
    foreach (i; 0 .. 4 * lat.nsites)
        lat.psi[i] = C(normFactor, 0) * lat.psi[i];

    // Check initial norm = 1
    double initNorm = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        auto z = lat.psi[i];
        initNorm += z.re * z.re + z.im * z.im;
    }
    assert(fabs(initNorm - 1.0) < 1e-12);

    // Apply one L-shift
    auto res = applyShift(lat, false, PAT_L, 1e-20);

    // Norm should be approximately preserved (some absorption at boundary)
    double postNorm = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        auto z = lat.psi[i];
        postNorm += z.re * z.re + z.im * z.im;
    }
    // Norm + absorbed should be close to 1. Small excess is from new sites
    // created during the shift that receive amplitude from boundary sites.
    double total = postNorm + res.probAbsorbed;
    assert(total > 0.95 && total < 1.05, "Norm + absorbed out of range");
}

unittest {
    import lattice : Lattice, DensityGrid, generateSites;

    // Coin preserves norm (theta=0 is identity)
    auto lat = Lattice.create(100000, 17);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    // Set psi to something nonzero
    foreach (n; 0 .. lat.nsites)
        lat.psi[4*n] = C(1.0 / sqrt(cast(double) lat.nsites), 0);

    double normBefore = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        auto z = lat.psi[i];
        normBefore += z.re * z.re + z.im * z.im;
    }

    // Apply coin with theta=0.5
    applyCoin(lat, true, cos(0.5), sin(0.5), 1);

    double normAfter = 0;
    foreach (i; 0 .. 4 * lat.nsites) {
        auto z = lat.psi[i];
        normAfter += z.re * z.re + z.im * z.im;
    }

    assert(fabs(normAfter - normBefore) < 1e-10, "Coin should preserve norm");
}
