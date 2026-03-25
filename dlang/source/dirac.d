/**
 * dirac.d — Dirac algebra, τ operators, and frame transport.
 *
 * Provides 4×4 complex matrix operations, the Dirac representation of
 * α and β matrices, τ operator construction, P± projectors, and the
 * frame transport unitary that maps between τ eigenspaces at neighboring sites.
 */
module dirac;

import std.math : sqrt, fabs;
import std.complex : Complex, complex;
import geometry : Vec3, tetDirs;

alias C = Complex!double;

C conj(C z) { return Complex!double(z.re, -z.im); }

/// 4×4 complex matrix stored as row-major flat array.
struct Mat4 {
    C[16] data;

    this(double _) { data[] = C(0, 0); }

    ref C opIndex(int r, int c) return {
        return data[4 * r + c];
    }

    C opIndex(int r, int c) const {
        return data[4 * r + c];
    }

    static Mat4 zero() { return Mat4(0); }

    static Mat4 eye() {
        auto m = Mat4(0);
        foreach (i; 0 .. 4) m[i, i] = C(1, 0);
        return m;
    }
}

/// C = A * B
Mat4 mul(Mat4 A, Mat4 B) {
    auto res = Mat4(0);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            C s = C(0, 0);
            foreach (k; 0 .. 4)
                s = s + A[i, k] * B[k, j];
            res[i, j] = s;
        }
    return res;
}

/// Trace of a 4×4 matrix.
C trace(Mat4 M) {
    C s = C(0, 0);
    foreach (i; 0 .. 4)
        s = s + M[i, i];
    return s;
}

// ---- Dirac matrices (Dirac representation) ----

immutable C ii = C(0, 1);
immutable C one = C(1, 0);
immutable C neg = C(-1, 0);
immutable C nii = C(0, -1);

Mat4 alpha(int idx) {
    auto m = Mat4(0);
    final switch (idx) {
        case 0: // σ₁ off-diagonal
            m[0,3] = one; m[1,2] = one; m[2,1] = one; m[3,0] = one;
            break;
        case 1: // σ₂ off-diagonal
            m[0,3] = nii; m[1,2] = ii; m[2,1] = nii; m[3,0] = ii;
            break;
        case 2: // σ₃ off-diagonal
            m[0,2] = one; m[1,3] = neg; m[2,0] = one; m[3,1] = neg;
            break;
    }
    return m;
}

Mat4 beta() {
    auto m = Mat4(0);
    m[0,0] = one; m[1,1] = one; m[2,2] = neg; m[3,3] = neg;
    return m;
}

// ---- τ operators ----

/// Construct τ = (√7/4)β + (3/4)(d · α) for a direction vector d.
Mat4 makeTau(Vec3 d) {
    immutable double nu = sqrt(7.0) / 4.0;
    auto tau = Mat4(0);
    // β contribution
    foreach (i; 0 .. 4)
        tau[i, i] = C(nu * (i < 2 ? 1.0 : -1.0), 0);
    // α contributions
    double[3] dd = [d.x, d.y, d.z];
    foreach (a; 0 .. 3) {
        auto al = alpha(a);
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4)
                tau[i, j] = tau[i, j] + C(0.75 * dd[a], 0) * al[i, j];
    }
    return tau;
}

/// P+ projector: (I + τ) / 2
Mat4 projPlus(Mat4 tau) {
    auto p = Mat4.eye();
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4)
            p[i, j] = C(0.5, 0) * (p[i, j] + tau[i, j]);
    return p;
}

/// P- projector: (I - τ) / 2
Mat4 projMinus(Mat4 tau) {
    auto p = Mat4.eye();
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4)
            p[i, j] = C(0.5, 0) * (p[i, j] - tau[i, j]);
    return p;
}

// ---- Frame transport ----

/// Compute the frame transport unitary U that maps the eigenspaces of
/// tau_from to those of tau_to.
///
/// U = (I + tau_to * tau_from) / (2 cos(θ/2))
/// where cos θ = tr(tau_to * tau_from) / 4.
Mat4 frameTransport(Mat4 tauFrom, Mat4 tauTo) {
    Mat4 prod = mul(tauTo, tauFrom);
    double cosTheta = trace(prod).re / 4.0;
    double cosHalf = sqrt((1.0 + cosTheta) / 2.0);
    double scale = 1.0 / (2.0 * cosHalf);

    auto U = Mat4(0);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4)
            U[i, j] = C(scale, 0) * (C(i == j ? 1.0 : 0.0, 0) + prod[i, j]);
    return U;
}

// ---- C interface ----

export extern(C)
void make_tau_c(const(double)* dir, double* out_re, double* out_im) {
    Vec3 d = Vec3(dir[0], dir[1], dir[2]);
    Mat4 tau = makeTau(d);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            out_re[4*i+j] = tau[i, j].re;
            out_im[4*i+j] = tau[i, j].im;
        }
}

export extern(C)
void frame_transport_c(const(double)* tau_from_re, const(double)* tau_from_im,
                       const(double)* tau_to_re, const(double)* tau_to_im,
                       double* out_re, double* out_im) {
    auto tf = Mat4(0);
    auto tt = Mat4(0);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            int idx = 4*i+j;
            tf[i, j] = C(tau_from_re[idx], tau_from_im[idx]);
            tt[i, j] = C(tau_to_re[idx], tau_to_im[idx]);
        }
    Mat4 U = frameTransport(tf, tt);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            int idx = 4*i+j;
            out_re[idx] = U[i, j].re;
            out_im[idx] = U[i, j].im;
        }
}

// ---- D unit tests ----

unittest {
    // α matrices are Hermitian
    foreach (a; 0 .. 3)
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                auto diff = alpha(a)[i,j] - conj(alpha(a)[j,i]);
                assert(fabs(diff.re) < 1e-14 && fabs(diff.im) < 1e-14);
            }
}

unittest {
    // α_i² = I
    foreach (a; 0 .. 3) {
        Mat4 sq = mul(alpha(a), alpha(a));
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                double expected = (i == j) ? 1.0 : 0.0;
                assert(fabs(sq[i,j].re - expected) < 1e-14);
                assert(fabs(sq[i,j].im) < 1e-14);
            }
    }
}

unittest {
    // {α_i, α_j} = 0 for i ≠ j (Clifford anticommutation)
    foreach (a; 0 .. 3)
        foreach (b; a+1 .. 3) {
            Mat4 ab = mul(alpha(a), alpha(b));
            Mat4 ba = mul(alpha(b), alpha(a));
            foreach (i; 0 .. 4)
                foreach (j; 0 .. 4) {
                    auto s = ab[i,j] + ba[i,j];
                    assert(fabs(s.re) < 1e-14 && fabs(s.im) < 1e-14);
                }
        }
}

unittest {
    // τ² = I for all 4 tet directions
    foreach (a; 0 .. 4) {
        auto d = tetDirs[a];
        Mat4 tau = makeTau(Vec3(d.x, d.y, d.z));
        Mat4 sq = mul(tau, tau);
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                double expected = (i == j) ? 1.0 : 0.0;
                assert(fabs(sq[i,j].re - expected) < 1e-13);
                assert(fabs(sq[i,j].im) < 1e-13);
            }
    }
}

unittest {
    // τ is Hermitian for all 4 tet directions
    foreach (a; 0 .. 4) {
        auto d = tetDirs[a];
        Mat4 tau = makeTau(Vec3(d.x, d.y, d.z));
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                auto diff = tau[i,j] - conj(tau[j,i]);
                assert(fabs(diff.re) < 1e-14 && fabs(diff.im) < 1e-14);
            }
    }
}

unittest {
    // P+ + P- = I, P+² = P+
    auto d = tetDirs[2];
    Mat4 tau = makeTau(Vec3(d.x, d.y, d.z));
    Mat4 pp = projPlus(tau);
    Mat4 pm = projMinus(tau);

    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            auto s = pp[i,j] + pm[i,j];
            double expected = (i == j) ? 1.0 : 0.0;
            assert(fabs(s.re - expected) < 1e-14);
        }

    Mat4 pp2 = mul(pp, pp);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            auto diff = pp2[i,j] - pp[i,j];
            assert(fabs(diff.re) < 1e-13 && fabs(diff.im) < 1e-13);
        }
}

unittest {
    // Frame transport is unitary: U†U = I
    auto d0 = tetDirs[0];
    auto d1 = tetDirs[1];
    Mat4 t0 = makeTau(Vec3(d0.x, d0.y, d0.z));
    Mat4 t1 = makeTau(Vec3(d1.x, d1.y, d1.z));
    Mat4 U = frameTransport(t0, t1);

    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            C s = C(0, 0);
            foreach (k; 0 .. 4)
                s = s + conj(U[k,i]) * U[k,j];
            double expected = (i == j) ? 1.0 : 0.0;
            assert(fabs(s.re - expected) < 1e-12);
            assert(fabs(s.im) < 1e-12);
        }
}

unittest {
    // Frame transport intertwines: U τ_from = τ_to U
    auto d0 = tetDirs[0];
    auto d1 = tetDirs[1];
    Mat4 t0 = makeTau(Vec3(d0.x, d0.y, d0.z));
    Mat4 t1 = makeTau(Vec3(d1.x, d1.y, d1.z));
    Mat4 U = frameTransport(t0, t1);

    Mat4 lhs = mul(U, t0);
    Mat4 rhs = mul(t1, U);
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4) {
            auto diff = lhs[i,j] - rhs[i,j];
            assert(fabs(diff.re) < 1e-12);
            assert(fabs(diff.im) < 1e-12);
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
        auto sum = Mat4(0);
        foreach (a; 0 .. 4)
            foreach (i; 0 .. 4)
                foreach (j; 0 .. 4)
                    sum[i,j] = sum[i,j] + C(E[coord][a], 0) * taus[a][i,j];

        auto expected = alpha(coord);
        foreach (i; 0 .. 4)
            foreach (j; 0 .. 4) {
                auto diff = sum[i,j] - expected[i,j];
                assert(fabs(diff.re) < 1e-13);
                assert(fabs(diff.im) < 1e-13);
            }
    }
}
