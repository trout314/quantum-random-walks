#!/usr/bin/env python3
"""
Analyze the walk's tau operator at several sites to understand:
1. What is the P+/P- content of (1,0,0,0) under each site's tau?
2. How much does tau vary along the chain?
3. What fixed 4x4 Dirac IC would reproduce the walk's asymmetry?
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helix_geometry import build_taus, make_tau, exit_direction

# Dirac matrices (matching walk_1d.c)
alpha = np.zeros((3, 4, 4), dtype=complex)
alpha[0] = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
alpha[1] = [[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]]
alpha[2] = [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]]
beta = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)

PAT_R = [1, 3, 0, 2]
PAT_L = [0, 1, 2, 3]

def build_chain(pat, n_sites):
    """Build chain and return tau at each site."""
    taus_arr = build_taus(n_sites)
    taus = [taus_arr[i] for i in range(n_sites)]
    faces = [pat[i % 4] for i in range(n_sites)]
    return taus, faces

print("=" * 60)
print("Tau analysis along helix chain")
print("=" * 60)

chi = np.array([1, 0, 0, 0], dtype=complex)

for pat_name, pat in [('R {1,3,0,2}', PAT_R), ('L {0,1,2,3}', PAT_L)]:
    print(f"\n--- {pat_name} pattern ---")
    taus, faces = build_chain(pat, 16)

    for i in range(8):
        tau = taus[i]
        # Eigendecompose tau
        evals, evecs = np.linalg.eigh(tau)
        # tau has eigenvalues +1 and -1 (each 2x degenerate)
        # P+ projects onto +1 eigenspace, P- onto -1
        Pp = 0.5 * (np.eye(4) + tau)
        Pm = 0.5 * (np.eye(4) - tau)

        pp_chi = Pp @ chi
        pm_chi = Pm @ chi
        frac_p = np.real(np.conj(pp_chi) @ pp_chi)
        frac_m = np.real(np.conj(pm_chi) @ pm_chi)

        print(f"  site {i} (face {faces[i]}): P+={frac_p:.4f}, P-={frac_m:.4f}")

# Check: what does tau look like in the beta eigenbasis?
print("\n" + "=" * 60)
print("Structure of tau at site 0 (R-chain)")
print("=" * 60)
taus_R, _ = build_chain(PAT_R, 4)
tau0 = taus_R[0]
print(f"\ntau[0] =")
for row in tau0:
    print("  [" + ", ".join(f"{v.real:+.4f}" if abs(v.imag)<1e-10
                            else f"{v:+.4f}" for v in row) + "]")

# Decompose tau into beta and alpha components
beta_coeff = np.trace(tau0 @ beta) / 4  # = sqrt(7)/4
print(f"\nbeta coefficient: {beta_coeff.real:.6f} (expect sqrt(7)/4 = {np.sqrt(7)/4:.6f})")

for a in range(3):
    alpha_coeff = np.trace(tau0 @ alpha[a]) / 4
    print(f"alpha_{a+1} coefficient: {alpha_coeff.real:.6f}{'' if abs(alpha_coeff.imag)<1e-10 else f' + {alpha_coeff.imag:.6f}i'}")

# What IC in the standard Dirac (H = ck*alpha_1 + m*beta) gives 83/17 split?
print("\n" + "=" * 60)
print("Finding Dirac IC that matches walk's P+/P- split")
print("=" * 60)

# In standard Dirac, "right-mover" = positive energy = upper component of alpha_1 eigenstate
# alpha_1 eigenstates with eigenvalue +1: (1,0,0,1)/sqrt(2), (0,1,1,0)/sqrt(2)
# alpha_1 eigenstates with eigenvalue -1: (1,0,0,-1)/sqrt(2), (0,1,-1,0)/sqrt(2)
# beta eigenstates with eigenvalue +1: (1,0,0,0), (0,1,0,0)
# beta eigenstates with eigenvalue -1: (0,0,1,0), (0,0,0,1)

# For (1,0,0,0):
# - beta P+ fraction: 1.0, P- fraction: 0.0  (it IS a beta eigenstate)
# - alpha_1 P+ fraction: 0.5, P- fraction: 0.5  (50/50 in alpha_1 basis)

print("\n(1,0,0,0) projections:")
Pp_beta = 0.5 * (np.eye(4) + beta)
Pm_beta = 0.5 * (np.eye(4) - beta)
Pp_a1 = 0.5 * (np.eye(4) + alpha[0])
Pm_a1 = 0.5 * (np.eye(4) - alpha[0])

for name, Pp, Pm in [("beta", Pp_beta, Pm_beta), ("alpha_1", Pp_a1, Pm_a1)]:
    fp = np.real(np.conj(chi) @ Pp @ chi)
    fm = np.real(np.conj(chi) @ Pm @ chi)
    print(f"  {name}: P+={fp:.4f}, P-={fm:.4f}")

# The walk's tau[0] gives ~83% P+. What is the effective operator?
# tau = nu*beta + (3/4)*e.alpha where nu=sqrt(7)/4
# For (1,0,0,0): <chi|tau|chi> = nu*1 + (3/4)*e_z*<chi|alpha_3|chi>
# alpha_3 has <(1,0,0,0)|alpha_3|(1,0,0,0)> = 0
# So <chi|tau|chi> = nu = sqrt(7)/4 ≈ 0.661
# P+ fraction = (1 + <tau>)/2 = (1 + 0.661)/2 = 0.831 ✓

print(f"\n<(1,0,0,0)|tau[0]|(1,0,0,0)> = {np.real(np.conj(chi) @ tau0 @ chi):.6f}")
print(f"Expected: sqrt(7)/4 = {np.sqrt(7)/4:.6f}")
print(f"P+ = (1 + <tau>)/2 = {(1 + np.sqrt(7)/4)/2:.6f}")

# So the walk's 83% P+ comes entirely from the beta component of tau.
# In a Dirac equation with H = ck*A + m*B, the "right-mover" projector is P_A+ = (I+A)/2.
# To get 83% right-moving from (1,0,0,0), we need A such that <chi|A|chi> = sqrt(7)/4.
#
# One natural choice: A = tau itself (at some reference site).
# But tau changes site to site...
#
# Alternative: since <chi|tau|chi> = sqrt(7)/4 = <chi|beta|chi> for chi=(1,0,0,0),
# the splitting comes entirely from beta. So H = ck*beta + m*M would give the right split.

print("\n" + "=" * 60)
print("Key insight: for IC=(1,0,0,0), <tau> = <beta> = sqrt(7)/4")
print("The 83/17 split comes ENTIRELY from the beta part of tau.")
print("The e.alpha part contributes 0 because alpha has no diagonal elements.")
print("=" * 60)

# Let's check: does H = ck*beta give the right dispersion?
# beta^2 = I, so E = ±sqrt(c^2 k^2 + m^2) still holds
# beta eigenvalues are +1 (upper 2 comp) and -1 (lower 2 comp)
# So P+ for beta = upper two components, same as standard
# (1,0,0,0) is 100% beta P+... that's not 83%

# Wait - the walk P+ uses TAU, not beta. So even if H = ck*alpha_1 + m*beta,
# we need to compare the Dirac output using the WALK's tau projection, not the
# Dirac equation's natural basis.

print("\n" + "=" * 60)
print("Resolution: the Dirac P+/P- should be computed using tau, not beta or alpha_1")
print("The walk measures P± = (I ± tau)/2, so the Dirac comparison should too")
print("=" * 60)
