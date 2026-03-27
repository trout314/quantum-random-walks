/**
 * symmetry.d — A4 (chiral tetrahedral) symmetry diagnostic for the lattice.
 *
 * BFS from the origin, checking at each depth whether every site's 12
 * A4 images are present in the lattice.  Reports the depth at which
 * symmetry first breaks and summary statistics.
 */
module symmetry;

import std.stdio : stderr;
import geometry : Vec3, Mat3, buildAllA4Rotations;
import lattice : Lattice, ProximityGrid;

/// Run the BFS symmetry diagnostic.  Prints per-depth report to stderr.
void checkA4Symmetry(bool hasCoin)(ref Lattice!hasCoin lat, ref ProximityGrid grid, int maxDepth) {
    auto rots = buildAllA4Rotations();
    double tol = grid.cellSize * 0.5;  // search tolerance = dMin/2

    int ns = lat.nsites;
    auto depth = new int[ns];
    depth[] = -1;
    depth[0] = 0;

    int[] queue;
    queue.reserve(ns);
    queue ~= 0;
    int qHead = 0;

    // BFS to assign depths
    while (qHead < queue.length) {
        int s = queue[qHead++];
        int d = depth[s];
        if (d >= maxDepth) continue;

        // 4 chain neighbors: R-next, R-prev, L-next, L-prev
        foreach (isR; [true, false]) {
            if (lat.chainFace(s, isR) < 0) continue;
            foreach (dir; 0 .. 2) {
                int nb = (dir == 0) ? lat.chainNext(s, isR) : lat.chainPrev(s, isR);
                if (nb < 0 || depth[nb] >= 0) continue;
                depth[nb] = d + 1;
                queue ~= nb;
            }
        }
    }

    // Check symmetry at each depth
    stderr.writefln("\n--- A4 symmetry diagnostic (tol=%.4f) ---", tol);
    stderr.writefln("# depth  sites  symmetric  broken  frac_sym");

    int firstBroken = -1;

    foreach (d; 0 .. maxDepth + 1) {
        int nSites = 0;
        int nSym = 0;
        int nBroken = 0;

        foreach (s; 0 .. ns) {
            if (depth[s] != d) continue;
            nSites++;

            Vec3 pos = lat.sites[s].pos;
            bool allFound = true;

            // Check all 11 non-identity rotations
            foreach (ri; 1 .. 12) {
                Vec3 rp = rots[ri].apply(pos);
                int found = grid.findSiteNear(rp, tol);
                if (found < 0) {
                    allFound = false;
                    break;
                }
            }

            if (allFound) nSym++;
            else nBroken++;
        }

        if (nSites == 0) continue;

        double frac = cast(double) nSym / nSites;
        stderr.writefln("  %4d  %6d  %9d  %6d  %.4f", d, nSites, nSym, nBroken, frac);

        if (nBroken > 0 && firstBroken < 0)
            firstBroken = d;
    }

    if (firstBroken >= 0)
        stderr.writefln("First symmetry break at BFS depth %d", firstBroken);
    else
        stderr.writefln("All depths fully symmetric up to depth %d", maxDepth);
}

// ---- Unit tests (rotation tests now in geometry.d) ----
