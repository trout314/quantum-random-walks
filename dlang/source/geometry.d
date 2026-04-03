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

// ---- Analytic BC helix geometry ----
//
// All positions and directions computed from the closed-form vertex formula:
//   v_k = (r cos(kθ), r sin(kθ), k h)   (unit edge, formula frame)
// then transformed to the walk code frame via Procrustes alignment.
// No sequential reflections or reorthogonalization needed.

import std.math : cos, sin;

/// BC helix vertex parameters (unit edge length).
private enum double THETA_BC = 2.300523983021863;  // arccos(-2/3)
private enum double R_VERTEX = 0.5196152422706632;  // 3√3/10
private enum double H_VERTEX = 0.3162277660168379;  // 1/√10

/// Scale factor: unit-sphere tetrahedra / unit-edge tetrahedra.
private enum double HELIX_SCALE = 1.632993161855452;  // 2√6/3

/// Procrustes alignment rotation (row-major 3×3).
private immutable double[9] R_ALIGN = [
    +5.555555555553510e-01, -3.975231960002475e-01, -7.302967433402225e-01,
    +7.698003589194571e-01, -8.606629658278393e-02, +6.324555320336754e-01,
    -3.142696805278236e-01, -9.135468796040372e-01, +2.581988897471603e-01,
];

/// Procrustes alignment translation.
private immutable double[3] T_ALIGN = [
    +4.714045207910544e-01, -6.531972647421682e-01, -6.666666666666909e-02,
];

/// Apply the alignment transform: walk_pos = SCALE * R @ formula_pos + T.
private Vec3 alignToWalkFrame(Vec3 vf) {
    double sx = HELIX_SCALE * vf.x;
    double sy = HELIX_SCALE * vf.y;
    double sz = HELIX_SCALE * vf.z;
    return Vec3(
        R_ALIGN[0]*sx + R_ALIGN[1]*sy + R_ALIGN[2]*sz + T_ALIGN[0],
        R_ALIGN[3]*sx + R_ALIGN[4]*sy + R_ALIGN[5]*sz + T_ALIGN[1],
        R_ALIGN[6]*sx + R_ALIGN[7]*sy + R_ALIGN[8]*sz + T_ALIGN[2],
    );
}

/// Position of the k-th BC helix vertex in the walk code frame.
Vec3 helixVertex(int k) {
    double kd = cast(double) k;
    double angle = kd * THETA_BC;
    Vec3 vf = Vec3(R_VERTEX * cos(angle), R_VERTEX * sin(angle), kd * H_VERTEX);
    return alignToWalkFrame(vf);
}

/// Centroid (walker position) of the n-th tetrahedron.
/// Tetrahedron n has vertices {v_n, v_{n+1}, v_{n+2}, v_{n+3}}.
Vec3 helixCentroid(int n) {
    Vec3 c = Vec3(0, 0, 0);
    foreach (k; 0 .. 4)
        c = c + helixVertex(n + k);
    return c * 0.25;
}

/// Unit exit direction at site n.
/// This is the direction from the centroid toward the dropped vertex v_n
/// (the vertex not shared with tetrahedron n+1).
Vec3 helixExitDir(int n) {
    Vec3 c = helixCentroid(n);
    Vec3 v = helixVertex(n);
    Vec3 d = v - c;
    double nrm = norm(d);
    if (nrm > NORM_TOL)
        return d * (1.0 / nrm);
    return Vec3(0, 0, 1);
}

/// All 4 unit vertex directions from centroid of tetrahedron n.
Vec3[4] helixVertexDirs(int n) {
    Vec3 c = helixCentroid(n);
    Vec3[4] dirs;
    foreach (k; 0 .. 4) {
        Vec3 d = helixVertex(n + k) - c;
        double nrm = norm(d);
        if (nrm > NORM_TOL)
            dirs[k] = d * (1.0 / nrm);
        else
            dirs[k] = Vec3(0, 0, 0);
    }
    return dirs;
}

// ---- Parameterized helix for arbitrary chain origins (3D lattice) ----

/// Origin data for a BC helix chain starting at an arbitrary site.
/// Stores the rotation from the standard frame to the chain's local frame,
/// plus the starting position and chirality sign.
struct ChainOrigin {
    Mat3 rot;         /// rotation from standard frame to chain frame
    Vec3 pos0;        /// position of chain's first centroid
    double tSign;     /// +1 for R-helix, -1 for L-helix
    int[4] slotPerm;  /// walk_slot[i] = formula_vertex_offset slotPerm[i] at root
    int[4] facePat;   /// face pattern [1,3,0,2] for R or [0,2,1,3] for L
    int faceOffset;   /// root site's position in the face pattern cycle
}

/// Compute the chain origin from a site's position, directions, and chirality.
/// Finds the rotation R such that R maps the standard initial exit direction
/// to the chain's exit direction at its root, and R maps the standard
/// initial centroid displacement pattern to the chain's displacement pattern.
/// Cached standard reference directions for R and L chirality.
/// Computed on first use, then reused by all subsequent calls.
private bool _stdDirsComputed = false;
private Vec3[4] _stdDirsR, _stdDirsL;

private void ensureStdDirs() {
    if (_stdDirsComputed) return;
    foreach (chirality; 0 .. 2) {
        double tSign = (chirality == 0) ? 1.0 : -1.0;
        Vec3 c0 = Vec3(0, 0, 0);
        Vec3[4] vStd;
        foreach (k; 0 .. 4) {
            double kd = cast(double) k;
            Vec3 vf = Vec3(
                R_VERTEX * cos(kd * THETA_BC * tSign),
                R_VERTEX * sin(kd * THETA_BC * tSign),
                kd * H_VERTEX);
            vStd[k] = alignToWalkFrame(vf);
            c0 = c0 + vStd[k];
        }
        c0 = c0 * 0.25;
        Vec3[4]* target = (chirality == 0) ? &_stdDirsR : &_stdDirsL;
        foreach (k; 0 .. 4) {
            Vec3 d = vStd[k] - c0;
            double nrm = norm(d);
            if (nrm > NORM_TOL) (*target)[k] = d * (1.0 / nrm);
        }
    }
    _stdDirsComputed = true;
}

ChainOrigin computeChainOrigin(Vec3 pos, Vec3[4] dirs, int face, bool isR) {
    ChainOrigin origin;
    origin.pos0 = pos;
    origin.tSign = isR ? 1.0 : -1.0;

    // Use cached standard reference directions for this chirality.
    ensureStdDirs();
    Vec3[4] stdDirs = isR ? _stdDirsR : _stdDirsL;

    // Chain dirs (already unit vectors in dirs[])
    // We need to find the permutation + rotation that maps stdDirs → dirs.
    // Use the fact that all 4 directions sum to zero and have pairwise dot = -1/3.
    // The rotation R satisfies R · stdDirs[k] ≈ dirs[perm[k]].
    // For Kabsch, we need the correspondence. Use the exit direction to anchor:
    // stdDirs[0] is the exit dir in standard frame, dirs[face] is exit in chain frame.
    // Match stdDirs[0] → dirs[face], then find best R for the remaining 3.

    // Build correspondence: stdDirs[0] → dirs[face]
    // For the other 3 std dirs, find the best-matching chain dir by dot product.
    int[4] perm;
    perm[0] = face;
    bool[4] used;
    used[face] = true;

    foreach (si; 1 .. 4) {
        double bestDot = -2;
        int bestJ = -1;
        foreach (j; 0 .. 4) {
            if (used[j]) continue;
            double d = dot(stdDirs[si], dirs[j]); // approximate: dirs may be rotated
            // Actually we want the j that, under R, maps stdDirs[si] closest to dirs[j].
            // Since R maps stdDirs[0]→dirs[face], use the ANGULAR relationship:
            // dot(stdDirs[0], stdDirs[si]) should equal dot(dirs[face], dirs[j])
            // All pairs have dot = -1/3, so this doesn't disambiguate.
            // Use cross products instead: stdDirs[0] × stdDirs[si] should map to dirs[face] × dirs[j]
            // (up to rotation). Match by maximizing triple product consistency.

            // Simpler: just use greedy matching. Build a crude R from the exit direction
            // alignment, then use it to predict where stdDirs[si] maps, and pick closest.
            if (si == 1 && bestJ < 0) { bestJ = j; continue; } // need initial R
        }
        // Fall back to a different strategy: Kabsch on all matched pairs.
    }

    // Actually, simpler approach: build R from the 4×3 → 4×3 Kabsch directly,
    // trying all 24 permutations and picking the one with smallest residual.
    // But this is expensive. Since we only do it once per chain creation, it's OK.

    // Find R by constraining: stdDirs[0] must map to dirs[face] (exit direction).
    // Among all permutations with p[0] = face, compute R = (3/4) Σ dirs[p[k]] ⊗ stdDirs[k]
    // and pick the one that best reproduces the centroid displacement at site 1.
    //
    // For tetrahedral directions, ALL A4 rotations give zero residual on the 4 directions
    // (because E·Eᵀ = (4/3)I makes the Kabsch degenerate). The exit direction constraint
    // + centroid check resolves this ambiguity.

    double bestErr = 1e30;
    Mat3 bestR;

    static immutable int[4][24] allPerms = computeAllPerms();
    foreach (ref p; allPerms) {
        // Constraint: the exit direction must match
        if (p[0] != face) continue;

        // R_ij = (3/4) Σ_k dirs[p[k]]_i · stdDirs[k]_j
        Mat3 R;
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 3) {
                double s = 0;
                foreach (k; 0 .. 4) {
                    double di = void, sj = void;
                    final switch (i) { case 0: di=dirs[p[k]].x; break; case 1: di=dirs[p[k]].y; break; case 2: di=dirs[p[k]].z; break; }
                    final switch (j) { case 0: sj=stdDirs[k].x; break; case 1: sj=stdDirs[k].y; break; case 2: sj=stdDirs[k].z; break; }
                    s += di * sj;
                }
                R.m[3*i+j] = 0.75 * s;
            }

        // Check determinant — only proper rotations (det ≈ +1) are valid.
        double det = R.m[0]*(R.m[4]*R.m[8]-R.m[5]*R.m[7])
                   - R.m[1]*(R.m[3]*R.m[8]-R.m[5]*R.m[6])
                   + R.m[2]*(R.m[3]*R.m[7]-R.m[4]*R.m[6]);
        if (det < 0) continue;  // skip improper rotations

        // Disambiguate using centroid displacements to sites 1 and 2.
        double err = 0;
        foreach (site; 1 .. 3) {
            Vec3 stdDelta = helixCentroid(site) - helixCentroid(0);
            Vec3 chainDelta = R.apply(stdDelta);
            // The actual chain displacement: walk code gives known positions
            // For site 1: -(2/3) * dirs[face]
            // For site 2: need to reflect and step again — but we don't want reflections!
            // Instead, compare R · (helixExitDir(site)) with expected chain exit dir.
            Vec3 stdExitN = helixExitDir(site);
            Vec3 chainExitMapped = R.apply(stdExitN);
            // The mapped exit dir should be a unit tetrahedral direction at the chain site.
            // Use displacement check for site 1:
            Vec3 actualDelta = dirs[face] * (-STEP_LEN);
            if (site == 1) {
                Vec3 diff = chainDelta - actualDelta;
                err += diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
            }
        }

        if (err < bestErr) {
            bestErr = err;
            bestR = R;
        }
    }

    origin.rot = bestR;

    // Determine the slot permutation: slotPerm[walk_slot] = formula_vertex_offset.
    // At the root, walk slot i has the direction that best matches R · stdDirs[k]
    // for some k. We find k for each i.
    foreach (i; 0 .. 4) {
        Vec3 target = dirs[i];  // walk slot i direction
        double bestDot2 = -2;
        int bestK = 0;
        foreach (k; 0 .. 4) {
            Vec3 mapped = bestR.apply(stdDirs[k]);
            double d2 = dot(mapped, target);
            if (d2 > bestDot2) { bestDot2 = d2; bestK = k; }
        }
        origin.slotPerm[i] = bestK;  // walk slot i = formula vertex bestK
    }

    // Store face pattern and root face offset
    immutable int[4] patR = [1, 3, 0, 2];
    immutable int[4] patL = [0, 2, 1, 3];
    origin.facePat = isR ? patR : patL;
    // Find where the root face sits in the pattern cycle
    origin.faceOffset = 0;
    foreach (i; 0 .. 4)
        if (origin.facePat[i] == face) { origin.faceOffset = i; break; }
    return origin;
}

/// Helper: compute all 24 permutations of {0,1,2,3} at compile time.
private int[4][24] computeAllPerms() {
    int[4][24] result;
    int idx = 0;
    foreach (a; 0 .. 4)
        foreach (b; 0 .. 4) {
            if (b == a) continue;
            foreach (c; 0 .. 4) {
                if (c == a || c == b) continue;
                int d = 6 - a - b - c;
                result[idx++] = [a, b, c, d];
            }
        }
    return result;
}

/// Vertex position for a chain with given origin, at formula vertex index k.
private Vec3 chainHelixVertex(const ChainOrigin* origin, int k) {
    double kd = cast(double) k;
    double angle = kd * THETA_BC * origin.tSign;
    // Formula-frame vertex (unit edge)
    Vec3 vf = Vec3(R_VERTEX * cos(angle), R_VERTEX * sin(angle), kd * H_VERTEX);
    // Transform to standard walk frame
    Vec3 vStd = alignToWalkFrame(vf);
    // Subtract centroid(0) for THIS chirality to get displacement from origin.
    // Must use the chain's tSign, not the hardcoded R-helix helixCentroid(0).
    Vec3 c0Std = Vec3(0, 0, 0);
    foreach (j; 0 .. 4) {
        double jd = cast(double) j;
        double a0 = jd * THETA_BC * origin.tSign;
        Vec3 v0 = Vec3(R_VERTEX * cos(a0), R_VERTEX * sin(a0), jd * H_VERTEX);
        c0Std = c0Std + alignToWalkFrame(v0);
    }
    c0Std = c0Std * 0.25;
    Vec3 delta = vStd - c0Std;
    // Rotate to chain frame and translate
    return origin.rot.apply(delta) + origin.pos0;
}

/// Centroid at chain index n for a chain with given origin.
Vec3 chainCentroid(const ChainOrigin* origin, int n) {
    Vec3 c = Vec3(0, 0, 0);
    foreach (k; 0 .. 4)
        c = c + chainHelixVertex(origin, n + k);
    return c * 0.25;
}

/// Exit direction at chain index n for a chain with given origin.
Vec3 chainExitDir(const ChainOrigin* origin, int n) {
    Vec3 c = chainCentroid(origin, n);
    Vec3 v = chainHelixVertex(origin, n);
    Vec3 d = v - c;
    double nrm = norm(d);
    if (nrm > NORM_TOL)
        return d * (1.0 / nrm);
    return Vec3(0, 0, 1);
}

/// All 4 vertex directions at chain index n for a chain with given origin.
/// The exit direction (dropped vertex = formula vertex n) is placed in the
/// walk slot determined by the face pattern. The other 3 directions fill
/// the remaining slots. This ensures dirs[face] gives the correct exit
/// direction for τ computation.
Vec3[4] chainVertexDirs(const ChainOrigin* origin, int n) {
    Vec3 c = chainCentroid(origin, n);

    // Compute all 4 formula-ordered directions (k=0 is the exit vertex)
    Vec3[4] formulaDirs;
    foreach (k; 0 .. 4) {
        Vec3 d = chainHelixVertex(origin, n + k) - c;
        double nrm = norm(d);
        if (nrm > NORM_TOL)
            formulaDirs[k] = d * (1.0 / nrm);
    }

    // The exit direction is formulaDirs[0] (dropped vertex = formula vertex n).
    // It must go into walk slot = facePat[n % 4] (the face used at chain index n).
    // The face at chain index n cycles through facePat starting at faceOffset
    int patIdx = ((n + origin.faceOffset) % 4 + 4) % 4;  // safe modulo for negative n
    int exitSlot = origin.facePat[patIdx];

    Vec3[4] dirs;
    dirs[exitSlot] = formulaDirs[0];

    // Fill remaining 3 slots with the other 3 directions.
    // The exact assignment doesn't matter for τ (only the exit dir is used).
    // For coin computation, we need valid tetrahedral directions — any order works
    // since the coin uses perpendicular vectors to the exit direction.
    int fi = 1;
    foreach (slot; 0 .. 4) {
        if (slot == exitSlot) continue;
        dirs[slot] = formulaDirs[fi++];
    }

    return dirs;
}

unittest {
    // ChainOrigin at standard origin should match helixCentroid/helixExitDir
    Vec3[4] stdDirs;
    stdDirs[] = tetDirs[];
    auto origin = computeChainOrigin(Vec3(0,0,0), stdDirs, 1, true);
    foreach (n; 0 .. 10) {
        Vec3 c1 = chainCentroid(&origin, n);
        Vec3 c2 = helixCentroid(n);
        assert(norm(c1 - c2) < 1e-10, "chainCentroid mismatch at standard origin");
    }
}

unittest {
    // Analytic centroid matches reflection-based computation for first few sites
    Vec3 pos = Vec3(0, 0, 0);
    auto dirs = initTet();
    immutable int[4] pat = [1, 3, 0, 2];

    foreach (n; 0 .. 20) {
        Vec3 c = helixCentroid(n);
        assert(fabs(c.x - pos.x) < 1e-12, "centroid x mismatch");
        assert(fabs(c.y - pos.y) < 1e-12, "centroid y mismatch");
        assert(fabs(c.z - pos.z) < 1e-12, "centroid z mismatch");

        helixStep(pos, dirs, pat[n % 4]);
        if ((n + 1) % 4 == 0)
            reorth(dirs);
    }
}

unittest {
    // Exit direction at site 0 matches e_1 = (2√2/3, 0, -1/3) (face 1)
    Vec3 d = helixExitDir(0);
    assert(fabs(d.x - tetDirs[1].x) < 1e-10);
    assert(fabs(d.y - tetDirs[1].y) < 1e-10);
    assert(fabs(d.z - tetDirs[1].z) < 1e-10);
}

unittest {
    // Centroid-to-centroid displacement has magnitude 2/3
    foreach (n; 0 .. 10) {
        Vec3 c0 = helixCentroid(n);
        Vec3 c1 = helixCentroid(n + 1);
        double d = norm(c1 - c0);
        assert(fabs(d - STEP_LEN) < 1e-12);
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
