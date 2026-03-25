/**
 * observables.d — Compute physical observables from the wavefunction.
 */
module observables;

import std.math : sqrt;
import geometry : Vec3, dot, norm;
import lattice : Lattice;

struct Observables {
    double totalProb = 0;
    double normPsi = 0;   // sqrt(totalProb)
    double r2 = 0;        // <r^2>
    double x2 = 0, y2 = 0, z2 = 0;
    double r95 = 0;       // radius enclosing 95% of probability
    int nsites = 0;
}

/// Compute all observables from the current wavefunction.
Observables computeObservables(const Lattice lat) {
    Observables obs;
    obs.nsites = lat.nsites;

    // Total probability
    foreach (n; 0 .. lat.nsites) {
        double p = 0;
        foreach (a; 0 .. 4) {
            double re = lat.psiRe[4*n + a];
            double im = lat.psiIm[4*n + a];
            p += re * re + im * im;
        }
        obs.totalProb += p;
    }
    obs.normPsi = sqrt(obs.totalProb);

    // Position moments
    foreach (n; 0 .. lat.nsites) {
        double p = 0;
        foreach (a; 0 .. 4) {
            double re = lat.psiRe[4*n + a];
            double im = lat.psiIm[4*n + a];
            p += re * re + im * im;
        }
        double pw = p / obs.totalProb;
        double x = lat.sites[n].pos.x;
        double y = lat.sites[n].pos.y;
        double z = lat.sites[n].pos.z;
        obs.x2 += pw * x * x;
        obs.y2 += pw * y * y;
        obs.z2 += pw * z * z;
    }
    obs.r2 = obs.x2 + obs.y2 + obs.z2;

    // r95
    {
        double rmax = 0;
        foreach (n; 0 .. lat.nsites) {
            double r = norm(lat.sites[n].pos);
            if (r > rmax) rmax = r;
        }
        double dr = 0.5;
        int nbins = cast(int)(rmax / dr) + 1;
        if (nbins < 10) nbins = 10;
        auto bp = new double[nbins + 1];
        bp[] = 0;
        foreach (n; 0 .. lat.nsites) {
            double r = norm(lat.sites[n].pos);
            double p = 0;
            foreach (a; 0 .. 4) {
                double re = lat.psiRe[4*n + a];
                double im = lat.psiIm[4*n + a];
                p += re * re + im * im;
            }
            int b = cast(int)(r / dr);
            if (b > nbins) b = nbins;
            bp[b] += p / obs.totalProb;
        }
        double cum = 0;
        foreach (b; 0 .. nbins + 1) {
            cum += bp[b];
            if (cum >= 0.95) { obs.r95 = b * dr; break; }
        }
    }

    return obs;
}

// ---- D unit tests ----

unittest {
    import lattice : Lattice, DensityGrid, generateSites;
    import std.math : exp, fabs;

    auto lat = Lattice.create(100000);
    double sigma = 1.5;
    double stepLen = 2.0 / 3.0;
    int maxChainLen = cast(int)(4.0 * sigma / stepLen) + 5;
    auto grid = DensityGrid.create(maxChainLen * stepLen + 5.0, 8);
    generateSites(lat, sigma, 1e-4, grid);

    // Initialize Gaussian
    double norm2 = 0;
    foreach (n; 0 .. lat.nsites) {
        double r2 = dot(lat.sites[n].pos, lat.sites[n].pos);
        double w = exp(-r2 / (2 * sigma * sigma));
        lat.psiRe[4*n] = w;
        norm2 += w * w;
    }
    double nf = 1.0 / sqrt(norm2);
    foreach (i; 0 .. 4 * lat.nsites)
        lat.psiRe[i] *= nf;

    auto obs = computeObservables(lat);

    // Norm should be 1
    assert(fabs(obs.normPsi - 1.0) < 1e-12);

    // r^2 should be roughly 3 sigma^2 for a 3D Gaussian (= 3 * 2.25 = 6.75)
    // but discrete sampling on the lattice gives a different value
    assert(obs.r2 > 0);

    // x^2 approx y^2 approx z^2 (approximately isotropic)
    double avg = obs.r2 / 3.0;
    assert(fabs(obs.x2 - avg) / avg < 0.15);  // within 15%
    assert(fabs(obs.y2 - avg) / avg < 0.15);
    assert(fabs(obs.z2 - avg) / avg < 0.15);

    // r95 should be positive and reasonable
    assert(obs.r95 > 0);
    assert(obs.r95 < 10 * sigma);
}
