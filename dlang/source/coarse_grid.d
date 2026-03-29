/**
 * coarse_grid.d — Coarse-grained 3D grid for accumulating spinor amplitudes.
 *
 * Sites from the fine lattice are binned into cubic grid cells.
 * Spinor amplitudes (4-component complex) are summed within each cell,
 * enabling interference between different chain branches.
 */
module coarse_grid;

import std.math : sqrt, floor, fabs, exp;
import geometry : Vec3, dot;

struct CoarseGrid {
    double cellSize;
    int nCells;       // cells per dimension
    double halfExtent;

    // Storage: 4 complex components per cell, flattened as [ix][iy][iz][component]
    // Total: nCells^3 * 8 doubles (4 re + 4 im)
    double[] re;
    double[] im;

    static CoarseGrid create(double halfExtent, double cellSize) {
        CoarseGrid g;
        g.cellSize = cellSize;
        g.halfExtent = halfExtent;
        g.nCells = cast(int)(2.0 * halfExtent / cellSize) + 1;
        int totalCells = g.nCells * g.nCells * g.nCells;
        g.re = new double[4 * totalCells];
        g.im = new double[4 * totalCells];
        g.re[] = 0;
        g.im[] = 0;
        return g;
    }

    /// Convert position to cell index. Returns -1 if out of range.
    int cellIndex(Vec3 pos) const {
        int ix = cast(int) floor((pos.x + halfExtent) / cellSize);
        int iy = cast(int) floor((pos.y + halfExtent) / cellSize);
        int iz = cast(int) floor((pos.z + halfExtent) / cellSize);
        if (ix < 0 || ix >= nCells || iy < 0 || iy >= nCells || iz < 0 || iz >= nCells)
            return -1;
        return (ix * nCells + iy) * nCells + iz;
    }

    /// Add a 4-component spinor amplitude to the grid cell at position pos.
    void addAmplitude(Vec3 pos, const double* psiRe, const double* psiIm) {
        int ci = cellIndex(pos);
        if (ci < 0) return;
        foreach (a; 0 .. 4) {
            re[4 * ci + a] += psiRe[a];
            im[4 * ci + a] += psiIm[a];
        }
    }

    /// Compute total probability in the grid: Σ |ψ_cell|²
    double totalProb() const {
        double p = 0;
        foreach (i; 0 .. cast(int)(re.length))
            p += re[i] * re[i] + im[i] * im[i];
        return p;
    }

    /// Write the grid to a file. Format: ix iy iz x y z |ψ|² re0 im0 re1 im1 re2 im2 re3 im3
    /// Only writes cells with |ψ|² > threshold.
    void writeToFile(string filename, double threshold = 1e-15) const {
        import std.stdio : File;
        auto f = File(filename, "w");
        f.writefln("# CoarseGrid: cellSize=%.4f nCells=%d halfExtent=%.2f",
                   cellSize, nCells, halfExtent);
        f.writefln("# ix iy iz x y z prob re0 im0 re1 im1 re2 im2 re3 im3");

        int totalCells = nCells * nCells * nCells;
        foreach (ci; 0 .. totalCells) {
            double prob = 0;
            foreach (a; 0 .. 4)
                prob += re[4*ci+a]*re[4*ci+a] + im[4*ci+a]*im[4*ci+a];
            if (prob < threshold) continue;

            int iz = ci % nCells;
            int iy = (ci / nCells) % nCells;
            int ix = ci / (nCells * nCells);
            double x = (ix + 0.5) * cellSize - halfExtent;
            double y = (iy + 0.5) * cellSize - halfExtent;
            double z = (iz + 0.5) * cellSize - halfExtent;

            f.writef("%d %d %d %.4f %.4f %.4f %.6e", ix, iy, iz, x, y, z, prob);
            foreach (a; 0 .. 4)
                f.writef(" %.6e %.6e", re[4*ci+a], im[4*ci+a]);
            f.writeln();
        }
    }
}
