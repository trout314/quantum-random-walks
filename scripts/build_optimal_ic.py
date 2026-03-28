#!/usr/bin/env python3
"""
Build the walk-optimal IC from low-energy eigenstates and save to file.
The IC is a 4-component spinor at each site, stored as:
  site_offset re0 im0 re1 im1 re2 im2 re3 im3
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fourier_dispersion import build_walk_operator
from src.helix_geometry import build_taus, centroid

def build_optimal_ic(N_small=400, sigma=50, mixPhi=0.03, energy_cutoff=0.08,
                     outfile='/tmp/optimal_ic.dat'):
    pat = [1, 3, 0, 2]
    center = N_small // 2

    print(f"Building walk operator (N={N_small})...")
    taus = build_taus(N_small)
    W = build_walk_operator(N_small, taus, mixPhi)

    print("Diagonalizing...")
    eigenvalues, eigenvectors = np.linalg.eig(W)
    E = np.angle(eigenvalues)

    near_gap = np.abs(E) < energy_cutoff
    print(f"Near-gap states (|E|<{energy_cutoff}): {near_gap.sum()}")

    # Build localized wavepacket from near-gap states
    xs = np.arange(N_small) - center
    gauss = np.exp(-xs**2 / (2*sigma**2))

    coeffs = np.zeros(len(E), dtype=complex)
    for j in np.where(near_gap)[0]:
        psi_j = eigenvectors[:, j].reshape(N_small, 4)
        overlap = 0
        for n in range(N_small):
            for a in range(4):
                overlap += np.conj(psi_j[n, a]) * gauss[n]
        coeffs[j] = overlap

    psi0 = eigenvectors @ coeffs
    psi0 /= np.linalg.norm(psi0)

    # Extract per-site spinors
    prob = np.array([np.sum(np.abs(psi0[4*n:4*n+4])**2) for n in range(N_small)])
    xm = np.sum(prob * xs) / np.sum(prob)
    var = np.sum(prob * (xs - xm)**2) / np.sum(prob)
    print(f"Wavepacket: <x>={xm:.2f}, width={np.sqrt(var):.1f}")

    # Save: only sites with significant amplitude (within 4*sigma of center)
    margin = int(4 * sigma) + 10
    lo = max(0, center - margin)
    hi = min(N_small, center + margin + 1)

    with open(outfile, 'w') as f:
        f.write(f"# Walk-optimal IC: N={N_small}, sigma={sigma}, mixPhi={mixPhi}\n")
        f.write(f"# center={center}, energy_cutoff={energy_cutoff}\n")
        f.write(f"# site_offset re0 im0 re1 im1 re2 im2 re3 im3\n")
        for n in range(lo, hi):
            spinor = psi0[4*n:4*n+4]
            offset = n - center
            parts = [f"{offset}"]
            for a in range(4):
                parts.append(f"{spinor[a].real:.15e}")
                parts.append(f"{spinor[a].imag:.15e}")
            f.write(" ".join(parts) + "\n")

    print(f"Saved {hi-lo} sites to {outfile}")
    return psi0, center

if __name__ == '__main__':
    build_optimal_ic()
