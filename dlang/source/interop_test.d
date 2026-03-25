/**
 * walk.d — Quantum walk on tetrahedral lattice.
 *
 * Dummy roundtrip functions to verify D/Python interop via ctypes.
 * Will be replaced with the actual walk implementation.
 */
module interop_test;

import core.stdc.stdio : printf;

/// Scale every element of an array by a constant. Returns 0 on success.
export extern(C)
int scale_array(double* data, int n, double factor) {
    if (data is null || n <= 0) return -1;
    foreach (i; 0 .. n)
        data[i] *= factor;
    return 0;
}

/// Compute the dot product of two arrays. Result written to *out_result.
export extern(C)
int dot_product(const(double)* a, const(double)* b, int n, double* out_result) {
    if (a is null || b is null || out_result is null || n <= 0) return -1;
    double sum = 0;
    foreach (i; 0 .. n)
        sum += a[i] * b[i];
    *out_result = sum;
    return 0;
}

/// Apply a simple 2x2 matrix [[a,b],[c,d]] to each pair of elements.
/// Operates in-place on an array of length 2*n_pairs.
export extern(C)
int apply_matrix_2x2(double* data, int n_pairs,
                     double m00, double m01, double m10, double m11) {
    if (data is null || n_pairs <= 0) return -1;
    foreach (i; 0 .. n_pairs) {
        double x = data[2*i];
        double y = data[2*i + 1];
        data[2*i]     = m00*x + m01*y;
        data[2*i + 1] = m10*x + m11*y;
    }
    return 0;
}

/// Return a version string for sanity checking.
export extern(C)
const(char)* walk_version() {
    return "quantum-walk-d 0.1.0";
}
