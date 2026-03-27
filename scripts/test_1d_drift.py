#!/usr/bin/env python3
"""
Measure the RMS difference between the 1D walk and the continuum Dirac solver.

Uses beta-symmetric IC (1,0,1,0)/sqrt(2) with frame transport disabled (icType=1),
which isolates the lattice drift from Zitterbewegung.

Outputs a single RMS score: lower = better match to Dirac.
"""
import numpy as np
import subprocess, sys

sys.path.insert(0, '/home/aaron-trout/Desktop/quantum-random-walks/scripts')
from dirac_1d_4comp import solve_dirac_1d_4comp

WALK_BIN = '/home/aaron-trout/Desktop/quantum-random-walks/dlang/build/walk_1d_d'
SIGMA = 50.0
NSTEPS = 200
MIX_PHI = 0.03
CHI0 = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2)

def run_walk(nsteps=NSTEPS, sigma=SIGMA, mixPhi=MIX_PHI):
    """Run the 1D walk and return (times, x_mean, density_at_final)."""
    r = subprocess.run(
        [WALK_BIN, '0', str(sigma), str(nsteps), '3', '0', '0',
         str(mixPhi), '0', '1'],
        capture_output=True, text=True
    )

    times, xmean, x2 = [], [], []
    for line in r.stdout.strip().split('\n'):
        if line.startswith('#'): continue
        parts = line.split()
        if len(parts) >= 4:
            times.append(int(parts[0]))
            xmean.append(float(parts[2]))
            x2.append(float(parts[3]))

    # Parse final density
    x_list, rho_list = [], []
    with open('/tmp/walk_1d_density.dat') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3:
                x_list.append(int(parts[0]))
                rho_list.append(float(parts[2]))

    return (np.array(times), np.array(xmean), np.array(x2),
            np.array(x_list), np.array(rho_list))


def run_dirac(nsteps=NSTEPS, sigma=SIGMA, mixPhi=MIX_PHI):
    """Run the Dirac solver and return (times, x_mean, density_at_final)."""
    N = 4096
    x_d, times_d, dens_d = solve_dirac_1d_4comp(
        N, CHI0, sigma, c=1.0, m=mixPhi, t_max=nsteps, dt_output=1
    )

    dt, dxm, dx2 = [], [], []
    for i, t in enumerate(times_d):
        rho = dens_d[i]['total']
        total = np.sum(rho)
        xm = np.sum(x_d * rho) / total
        x2m = np.sum(x_d**2 * rho) / total
        dt.append(t)
        dxm.append(xm)
        dx2.append(x2m)

    rho_final = dens_d[-1]['total']
    return np.array(dt), np.array(dxm), np.array(dx2), x_d, rho_final


def compute_score():
    """Compute RMS difference between walk and Dirac.

    Returns dict with:
      rms_xmean:    RMS of x_mean difference over all output times
      rms_density:  RMS of density difference at final time (over shared x range)
      max_drift:    maximum |x_mean_walk - x_mean_dirac|
    """
    w_t, w_xm, w_x2, w_xd, w_rho = run_walk()
    d_t, d_xm, d_x2, d_xd, d_rho = run_dirac()

    # Interpolate Dirac x_mean to walk output times
    d_xm_interp = np.interp(w_t, d_t, d_xm)
    drift = w_xm - d_xm_interp
    rms_xmean = np.sqrt(np.mean(drift**2))
    max_drift = np.max(np.abs(drift))

    # Density comparison at final time: interpolate Dirac to walk's x grid
    d_rho_interp = np.interp(w_xd, d_xd, d_rho, left=0, right=0)
    # Normalize both
    w_rho_n = w_rho / np.sum(w_rho)
    d_rho_n = d_rho_interp / np.sum(d_rho_interp) if np.sum(d_rho_interp) > 0 else d_rho_interp
    rms_density = np.sqrt(np.mean((w_rho_n - d_rho_n)**2))

    # L1 (total variation) — fraction of probability in the wrong place
    l1_density = np.sum(np.abs(w_rho_n - d_rho_n))

    return {
        'rms_xmean': rms_xmean,
        'max_drift': max_drift,
        'rms_density': rms_density,
        'l1_density': l1_density,
    }


if __name__ == '__main__':
    score = compute_score()
    print(f"RMS ⟨x⟩ drift:    {score['rms_xmean']:.4f}")
    print(f"Max |drift|:       {score['max_drift']:.4f}")
    print(f"RMS density diff:  {score['rms_density']:.6f}")
    print(f"L1 density diff:   {score['l1_density']:.4f}  (fraction of prob misplaced)")
