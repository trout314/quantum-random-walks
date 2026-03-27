/**
 * geometry.d — 3D vector operations and tetrahedral lattice geometry.
 *
 * The tetrahedral lattice is built from 4 unit vectors pointing to the
 * vertices of a regular tetrahedron centered at the origin. A "helix step"
 * moves along one of these directions and reflects all directions through
 * the face opposite that vertex.
 */
module geometry;

import std.math : sqrt, fabs;

/// BC helix step length: each step displaces by (2/3) of a tetrahedral direction.
enum double STEP_LEN = 2.0 / 3.0;

/// Re-orthogonalize direction frame every this many helix steps to correct FP drift.
enum int REORTH_INTERVAL = 8;

/// Tolerance for safe normalization (avoid division by near-zero).
enum double NORM_TOL = 1e-15;

struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3 opBinary(string op)(Vec3 rhs) const if (op == "+") {
        return Vec3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    Vec3 opBinary(string op)(Vec3 rhs) const if (op == "-") {
        return Vec3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    Vec3 opBinary(string op)(double s) const if (op == "*") {
        return Vec3(x * s, y * s, z * s);
    }

    Vec3 opBinaryRight(string op)(double s) const if (op == "*") {
        return Vec3(s * x, s * y, s * z);
    }
}

double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(Vec3 a) {
    return sqrt(dot(a, a));
}

/// Reflect v through the plane perpendicular to n (assumes |n|=1).
Vec3 reflect(Vec3 v, Vec3 n) {
    double d = 2.0 * dot(v, n);
    return Vec3(v.x - d * n.x, v.y - d * n.y, v.z - d * n.z);
}

/// The 4 tetrahedral unit vectors.
immutable Vec3[4] tetDirs = [
    Vec3(0, 0, 1),
    Vec3(2.0 * sqrt(2.0) / 3.0, 0, -1.0 / 3.0),
    Vec3(-sqrt(2.0) / 3.0, sqrt(6.0) / 3.0, -1.0 / 3.0),
    Vec3(-sqrt(2.0) / 3.0, -sqrt(6.0) / 3.0, -1.0 / 3.0),
];

// ---- A4 (chiral tetrahedral) rotation matrices ----

/// 3×3 rotation matrix, row-major.
struct Mat3 {
    double[9] m = [1,0,0, 0,1,0, 0,0,1];

    Vec3 apply(Vec3 v) const {
        return Vec3(
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z,
        );
    }

    /// Apply the transpose (= inverse for rotation matrices).
    Vec3 applyTranspose(Vec3 v) const {
        return Vec3(
            m[0]*v.x + m[3]*v.y + m[6]*v.z,
            m[1]*v.x + m[4]*v.y + m[7]*v.z,
            m[2]*v.x + m[5]*v.y + m[8]*v.z,
        );
    }
}

/// The 12 even permutations of {0,1,2,3} — the A4 group acting on tet vertices.
immutable int[4][12] A4_PERMS = [
    [0,1,2,3],  // identity
    [1,2,0,3],  // (012)
    [2,0,1,3],  // (021)
    [1,3,2,0],  // (013)
    [3,0,2,1],  // (031)
    [2,1,3,0],  // (023)
    [3,1,0,2],  // (032)
    [0,2,3,1],  // (123)
    [0,3,1,2],  // (132)
    [1,0,3,2],  // (01)(23)
    [2,3,0,1],  // (02)(13)
    [3,2,1,0],  // (03)(12)
];

/// Build the 3×3 rotation matrix for an A4 element given as a permutation
/// of the four tet directions.  R = (3/4) · E_σ · Eᵀ.
Mat3 buildA4Rotation(const int[4] perm) {
    Mat3 R;
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 3) {
            double s = 0;
            foreach (k; 0 .. 4) {
                double epi = 0, ekj = 0;
                final switch (i) {
                    case 0: epi = tetDirs[perm[k]].x; break;
                    case 1: epi = tetDirs[perm[k]].y; break;
                    case 2: epi = tetDirs[perm[k]].z; break;
                }
                final switch (j) {
                    case 0: ekj = tetDirs[k].x; break;
                    case 1: ekj = tetDirs[k].y; break;
                    case 2: ekj = tetDirs[k].z; break;
                }
                s += epi * ekj;
            }
            R.m[3*i + j] = s * 0.75;
        }
    return R;
}

/// Precompute all 12 A4 rotation matrices.
Mat3[12] buildAllA4Rotations() {
    Mat3[12] rots;
    foreach (i; 0 .. 12)
        rots[i] = buildA4Rotation(A4_PERMS[i]);
    return rots;
}

/// Initialize a mutable copy of the tetrahedral directions.
Vec3[4] initTet() {
    Vec3[4] d;
    d[] = tetDirs[];
    return d;
}

/// Take one BC helix step: translate by -2/3 * d[face], then reflect
/// all 4 direction vectors through d[face].
void helixStep(ref Vec3 pos, ref Vec3[4] dirs, int face) {
    Vec3 e = dirs[face];
    pos = pos + e * (-STEP_LEN);
    foreach (ref d; dirs)
        d = reflect(d, e);
}

/// Re-orthogonalize the 4 direction vectors by subtracting their mean
/// and normalizing each to unit length. Corrects for floating-point drift
/// after many helix steps.
void reorth(ref Vec3[4] dirs) {
    Vec3 m = Vec3(0, 0, 0);
    foreach (d; dirs)
        m = m + d;
    m = m * 0.25;
    foreach (ref d; dirs) {
        d = d - m;
        double n = norm(d);
        if (n > NORM_TOL)
            d = d * (1.0 / n);
    }
}

// ---- C interface for Python interop ----

export extern(C)
void tet_dirs(double* out_dirs) {
    foreach (i; 0 .. 4) {
        out_dirs[3 * i + 0] = tetDirs[i].x;
        out_dirs[3 * i + 1] = tetDirs[i].y;
        out_dirs[3 * i + 2] = tetDirs[i].z;
    }
}

export extern(C)
void helix_step_c(double* pos, double* dirs, int face) {
    Vec3 p = Vec3(pos[0], pos[1], pos[2]);
    Vec3[4] d;
    foreach (i; 0 .. 4)
        d[i] = Vec3(dirs[3*i], dirs[3*i+1], dirs[3*i+2]);
    helixStep(p, d, face);
    pos[0] = p.x; pos[1] = p.y; pos[2] = p.z;
    foreach (i; 0 .. 4) {
        dirs[3*i]   = d[i].x;
        dirs[3*i+1] = d[i].y;
        dirs[3*i+2] = d[i].z;
    }
}

// ---- D unit tests ----

unittest {
    // Tet directions are unit vectors
    foreach (d; tetDirs)
        assert(fabs(norm(d) - 1.0) < 1e-14);
}

unittest {
    // Centroid at origin: sum of tet directions = 0
    Vec3 s = Vec3(0, 0, 0);
    foreach (d; tetDirs)
        s = s + d;
    assert(fabs(norm(s)) < 1e-14);
}

unittest {
    // All pairwise dot products = -1/3
    foreach (i; 0 .. 4)
        foreach (j; i + 1 .. 4)
            assert(fabs(dot(tetDirs[i], tetDirs[j]) + 1.0 / 3.0) < 1e-14);
}

unittest {
    // Isotropy: E * E^T = (4/3) * I_3
    // where E is the 3x4 matrix of tet directions as columns
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 3) {
            double s = 0;
            double[4][3] E;
            foreach (k; 0 .. 4) {
                E[0][k] = tetDirs[k].x;
                E[1][k] = tetDirs[k].y;
                E[2][k] = tetDirs[k].z;
            }
            foreach (k; 0 .. 4)
                s += E[i][k] * E[j][k];
            double expected = (i == j) ? 4.0 / 3.0 : 0.0;
            assert(fabs(s - expected) < 1e-14);
        }
}

unittest {
    // Helix step and back returns to original position
    Vec3 pos = Vec3(0, 0, 0);
    auto dirs = initTet();
    Vec3 pos0 = pos;
    auto dirs0 = dirs;

    helixStep(pos, dirs, 0);
    // After one step, position moved
    assert(norm(pos - pos0) > 0.1);
    // Step back through same face returns to origin
    helixStep(pos, dirs, 0);
    assert(fabs(pos.x - pos0.x) < 1e-14);
    assert(fabs(pos.y - pos0.y) < 1e-14);
    assert(fabs(pos.z - pos0.z) < 1e-14);
}

unittest {
    // After reflection, directions still form a valid tetrahedron
    auto dirs = initTet();
    Vec3 pos = Vec3(0, 0, 0);
    helixStep(pos, dirs, 1);

    // Still unit vectors
    foreach (d; dirs)
        assert(fabs(norm(d) - 1.0) < 1e-14);
    // Still sum to zero
    Vec3 s = Vec3(0, 0, 0);
    foreach (d; dirs)
        s = s + d;
    assert(norm(s) < 1e-14);
    // Still have pairwise dot = -1/3
    foreach (i; 0 .. 4)
        foreach (j; i + 1 .. 4)
            assert(fabs(dot(dirs[i], dirs[j]) + 1.0 / 3.0) < 1e-14);
}

unittest {
    // Step length is 2/3
    Vec3 pos = Vec3(0, 0, 0);
    auto dirs = initTet();
    helixStep(pos, dirs, 0);
    assert(fabs(norm(pos) - 2.0 / 3.0) < 1e-14);
}

unittest {
    // Reorth corrects drift
    auto dirs = initTet();
    // Perturb slightly
    dirs[0].x += 0.01;
    dirs[1].y -= 0.005;
    reorth(dirs);
    // Should be close to unit vectors summing to zero
    foreach (d; dirs)
        assert(fabs(norm(d) - 1.0) < 0.02);
    Vec3 s = Vec3(0, 0, 0);
    foreach (d; dirs)
        s = s + d;
    assert(norm(s) < 0.02);
}

// ---- A4 rotation unit tests ----

unittest {
    // Identity rotation is the identity matrix
    auto rots = buildAllA4Rotations();
    foreach (i; 0 .. 3)
        foreach (j; 0 .. 3) {
            double expected = (i == j) ? 1.0 : 0.0;
            assert(fabs(rots[0].m[3*i+j] - expected) < 1e-12);
        }
}

unittest {
    // Every A4 rotation maps each tet direction to the permuted tet direction
    auto rots = buildAllA4Rotations();
    foreach (ri; 0 .. 12)
        foreach (k; 0 .. 4) {
            Vec3 re = rots[ri].apply(tetDirs[k]);
            Vec3 expected = tetDirs[A4_PERMS[ri][k]];
            assert(norm(re - expected) < 1e-12);
        }
}

unittest {
    // All A4 rotations are orthogonal (R^T R = I)
    auto rots = buildAllA4Rotations();
    foreach (ri; 0 .. 12)
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 3) {
                double s = 0;
                foreach (k; 0 .. 3)
                    s += rots[ri].m[3*k+i] * rots[ri].m[3*k+j];
                double expected = (i == j) ? 1.0 : 0.0;
                assert(fabs(s - expected) < 1e-12);
            }
}
