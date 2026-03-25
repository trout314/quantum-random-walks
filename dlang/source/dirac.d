/**
 * dirac.d — Dirac algebra, τ operators, and frame transport.
 *
 * Uses split real/imaginary storage (double[16] each) for 4×4 matrices
 * to enable efficient vectorization without Complex!double overhead.
 */
module dirac;

import std.math : sqrt, fabs;
import geometry : Vec3, tetDirs;

/// 4×4 complex matrix in split real/imaginary format.
struct Mat4 {
    double[16] re = 0;
    double[16] im = 0;

    static Mat4 eye() {
        Mat4 m;
        foreach (i; 0 .. 4) m.re[5*i] = 1.0;  // diagonal: 0,5,10,15
        return m;
    }

    /// Element access (for readability, not hot path).
    void set(int r, int c, double rval, double ival) {
        re[4*r+c] = rval;
        im[4*r+c] = ival;
    }
    double getRe(int r, int c) const { return re[4*r+c]; }
    double getIm(int r, int c) const { return im[4*r+c]; }
}

/// C = A * B (4×4 complex matrix multiply)
Mat4 mul(Mat4 A, Mat4 B) {
    Mat4 C;
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            double sr = 0, si = 0;
            foreach (k; 0 .. 4) {
                int ik = 4*i+k, kj = 4*k+j;
                sr += A.re[ik] * B.re[kj] - A.im[ik] * B.im[kj];
                si += A.re[ik] * B.im[kj] + A.im[ik] * B.re[kj];
            }
            C.re[4*i+j] = sr;
            C.im[4*i+j] = si;
        }
    return C;
}

/// Real part of trace.
double traceRe(Mat4 M) {
    return M.re[0] + M.re[5] + M.re[10] + M.re[15];
}

// ---- Dirac matrices ----

Mat4 alpha(int idx) {
    Mat4 m;
    final switch (idx) {
        case 0: // σ₁ off-diagonal
            m.re[3] = 1; m.re[6] = 1; m.re[9] = 1; m.re[12] = 1;
            break;
        case 1: // σ₂ off-diagonal
            m.im[3] = -1; m.im[6] = 1; m.im[9] = -1; m.im[12] = 1;
            break;
        case 2: // σ₃ off-diagonal
            m.re[2] = 1; m.re[7] = -1; m.re[8] = 1; m.re[13] = -1;
            break;
    }
    return m;
}

// ---- τ operators ----

/// Construct τ = (√7/4)β + (3/4)(d · α) for direction vector d.
Mat4 makeTau(Vec3 d) {
    enum double nu = sqrt(7.0) / 4.0;
    Mat4 tau;
    // β contribution: diag(ν, ν, -ν, -ν)
    tau.re[0] = nu; tau.re[5] = nu; tau.re[10] = -nu; tau.re[15] = -nu;
    // α contributions
    double[3] dd = [d.x, d.y, d.z];
    foreach (a; 0 .. 3) {
        auto al = alpha(a);
        double c = 0.75 * dd[a];
        foreach (idx; 0 .. 16) {
            tau.re[idx] += c * al.re[idx];
            tau.im[idx] += c * al.im[idx];
        }
    }
    return tau;
}

/// P+ projector: (I + τ) / 2
Mat4 projPlus(Mat4 tau) {
    Mat4 p;
    foreach (idx; 0 .. 16) {
        p.re[idx] = 0.5 * tau.re[idx];
        p.im[idx] = 0.5 * tau.im[idx];
    }
    p.re[0] += 0.5; p.re[5] += 0.5; p.re[10] += 0.5; p.re[15] += 0.5;
    return p;
}

/// P- projector: (I - τ) / 2
Mat4 projMinus(Mat4 tau) {
    Mat4 p;
    foreach (idx; 0 .. 16) {
        p.re[idx] = -0.5 * tau.re[idx];
        p.im[idx] = -0.5 * tau.im[idx];
    }
    p.re[0] += 0.5; p.re[5] += 0.5; p.re[10] += 0.5; p.re[15] += 0.5;
    return p;
}

// ---- Frame transport ----

/// U = (I + tau_to * tau_from) / (2 cos(θ/2))
Mat4 frameTransport(Mat4 tauFrom, Mat4 tauTo) {
    Mat4 prod = mul(tauTo, tauFrom);
    double cosTheta = traceRe(prod) / 4.0;
    double cosHalf = sqrt((1.0 + cosTheta) / 2.0);
    double scale = 1.0 / (2.0 * cosHalf);

    Mat4 U;
    foreach (idx; 0 .. 16) {
        U.re[idx] = scale * prod.re[idx];
        U.im[idx] = scale * prod.im[idx];
    }
    // Add scale * I
    U.re[0] += scale; U.re[5] += scale; U.re[10] += scale; U.re[15] += scale;
    return U;
}

// ---- Spinor operations (hot path) ----

/// Apply Mat4 to a spinor (4 complex values stored as 8 doubles: re0,im0,re1,im1,...).
/// psi_base points to the start of the site's 8 doubles.
void matVecSplit(ref const Mat4 M, const double* psi_re, const double* psi_im,
                 double* out_re, double* out_im) {
    foreach (a; 0 .. 4) {
        double sr = 0, si = 0;
        foreach (b; 0 .. 4) {
            int ab = 4*a+b;
            sr += M.re[ab] * psi_re[b] - M.im[ab] * psi_im[b];
            si += M.re[ab] * psi_im[b] + M.im[ab] * psi_re[b];
        }
        out_re[a] = sr;
        out_im[a] = si;
    }
}

// ---- C interface ----

export extern(C)
void make_tau_c(const(double)* dir, double* out_re, double* out_im) {
    Vec3 d = Vec3(dir[0], dir[1], dir[2]);
    Mat4 tau = makeTau(d);
    out_re[0..16] = tau.re[];
    out_im[0..16] = tau.im[];
}

export extern(C)
void frame_transport_c(const(double)* tau_from_re, const(double)* tau_from_im,
                       const(double)* tau_to_re, const(double)* tau_to_im,
                       double* out_re, double* out_im) {
    Mat4 tf, tt;
    tf.re[] = tau_from_re[0..16];
    tf.im[] = tau_from_im[0..16];
    tt.re[] = tau_to_re[0..16];
    tt.im[] = tau_to_im[0..16];
    Mat4 U = frameTransport(tf, tt);
    out_re[0..16] = U.re[];
    out_im[0..16] = U.im[];
}

// ---- D unit tests ----

unittest {
    // α matrices are Hermitian
    foreach (a; 0 .. 3)
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                auto al = alpha(a);
                // Hermitian: M[i,j] = conj(M[j,i])
                assert(fabs(al.getRe(i,j) - al.getRe(j,i)) < 1e-14);
                assert(fabs(al.getIm(i,j) + al.getIm(j,i)) < 1e-14);
            }
}

unittest {
    // α_i² = I
    foreach (a; 0 .. 3) {
        Mat4 sq = mul(alpha(a), alpha(a));
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                double expected = (i == j) ? 1.0 : 0.0;
                assert(fabs(sq.getRe(i,j) - expected) < 1e-14);
                assert(fabs(sq.getIm(i,j)) < 1e-14);
            }
    }
}

unittest {
    // {α_i, α_j} = 0 for i ≠ j
    foreach (a; 0 .. 3)
        foreach (b; a+1 .. 3) {
            Mat4 ab = mul(alpha(a), alpha(b));
            Mat4 ba = mul(alpha(b), alpha(a));
            foreach (idx; 0 .. 16) {
                assert(fabs(ab.re[idx] + ba.re[idx]) < 1e-14);
                assert(fabs(ab.im[idx] + ba.im[idx]) < 1e-14);
            }
        }
}

unittest {
    // τ² = I for all 4 tet directions
    foreach (a; 0 .. 4) {
        Mat4 tau = makeTau(Vec3(tetDirs[a].x, tetDirs[a].y, tetDirs[a].z));
        Mat4 sq = mul(tau, tau);
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                double expected = (i == j) ? 1.0 : 0.0;
                assert(fabs(sq.getRe(i,j) - expected) < 1e-13);
                assert(fabs(sq.getIm(i,j)) < 1e-13);
            }
    }
}

unittest {
    // τ is Hermitian
    foreach (a; 0 .. 4) {
        Mat4 tau = makeTau(Vec3(tetDirs[a].x, tetDirs[a].y, tetDirs[a].z));
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                assert(fabs(tau.getRe(i,j) - tau.getRe(j,i)) < 1e-14);
                assert(fabs(tau.getIm(i,j) + tau.getIm(j,i)) < 1e-14);
            }
    }
}

unittest {
    // P+ + P- = I, P+² = P+
    auto tau = makeTau(Vec3(tetDirs[2].x, tetDirs[2].y, tetDirs[2].z));
    auto pp = projPlus(tau);
    auto pm = projMinus(tau);

    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            int idx = 4*i+j;
            double expected = (i == j) ? 1.0 : 0.0;
            assert(fabs(pp.re[idx] + pm.re[idx] - expected) < 1e-14);
        }

    auto pp2 = mul(pp, pp);
    foreach (idx; 0 .. 16) {
        assert(fabs(pp2.re[idx] - pp.re[idx]) < 1e-13);
        assert(fabs(pp2.im[idx] - pp.im[idx]) < 1e-13);
    }
}

unittest {
    // Frame transport is unitary: U†U = I
    auto t0 = makeTau(Vec3(tetDirs[0].x, tetDirs[0].y, tetDirs[0].z));
    auto t1 = makeTau(Vec3(tetDirs[1].x, tetDirs[1].y, tetDirs[1].z));
    Mat4 U = frameTransport(t0, t1);

    // Compute U†U
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            double sr = 0, si = 0;
            foreach (k; 0 .. 4) {
                // U†[k,i] = conj(U[i,k])
                double uRe = U.getRe(k,i);
                double uIm = -U.getIm(k,i); // conjugate
                sr += uRe * U.getRe(k,j) - uIm * U.getIm(k,j);
                si += uRe * U.getIm(k,j) + uIm * U.getRe(k,j);
            }
            double expected = (i == j) ? 1.0 : 0.0;
            assert(fabs(sr - expected) < 1e-12);
            assert(fabs(si) < 1e-12);
        }
}

unittest {
    // Frame transport intertwines: U τ_from = τ_to U
    auto t0 = makeTau(Vec3(tetDirs[0].x, tetDirs[0].y, tetDirs[0].z));
    auto t1 = makeTau(Vec3(tetDirs[1].x, tetDirs[1].y, tetDirs[1].z));
    Mat4 U = frameTransport(t0, t1);
    Mat4 lhs = mul(U, t0);
    Mat4 rhs = mul(t1, U);
    foreach (idx; 0 .. 16) {
        assert(fabs(lhs.re[idx] - rhs.re[idx]) < 1e-12);
        assert(fabs(lhs.im[idx] - rhs.im[idx]) < 1e-12);
    }
}

unittest {
    // Dirac correspondence: Σ_a e_a^i τ_a = α_i
    Mat4[4] taus;
    foreach (a; 0 .. 4)
        taus[a] = makeTau(Vec3(tetDirs[a].x, tetDirs[a].y, tetDirs[a].z));

    double[4][3] E;
    foreach (a; 0 .. 4) {
        E[0][a] = tetDirs[a].x;
        E[1][a] = tetDirs[a].y;
        E[2][a] = tetDirs[a].z;
    }

    foreach (coord; 0 .. 3) {
        Mat4 sum;
        foreach (a; 0 .. 4)
            foreach (idx; 0 .. 16) {
                sum.re[idx] += E[coord][a] * taus[a].re[idx];
                sum.im[idx] += E[coord][a] * taus[a].im[idx];
            }

        auto expected = alpha(coord);
        foreach (idx; 0 .. 16) {
            assert(fabs(sum.re[idx] - expected.re[idx]) < 1e-13);
            assert(fabs(sum.im[idx] - expected.im[idx]) < 1e-13);
        }
    }
}
