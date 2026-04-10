#!/usr/bin/env python3
"""
Test D3: CPT symmetry in the long-wavelength (continuum) limit.

The full CPT residual is O(1), but CPT should be restored for
low-energy eigenstates where the lattice structure averages out.

Strategy: diagonalize W, restrict to the subspace |E| < E_cut,
and measure the CPT residual in that subspace.

Also test: how the residual scales with N (lattice refinement).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.helix_geometry import build_taus

# ---- Dirac algebra ----
I4 = np.eye(4, dtype=complex)
I2 = np.eye(2, dtype=complex)
Z2 = np.zeros((2,2), dtype=complex)
sigma = [
    np.array([[0,1],[1,0]], dtype=complex),
    np.array([[0,-1j],[1j,0]], dtype=complex),
    np.array([[1,0],[0,-1]], dtype=complex),
]
def blk(a,b,c,d): return np.block([[a,b],[c,d]])

gamma0 = blk(I2,Z2,Z2,-I2)
gamma1 = blk(Z2,sigma[0],-sigma[0],Z2)
gamma2 = blk(Z2,sigma[1],-sigma[1],Z2)
gamma3 = blk(Z2,sigma[2],-sigma[2],Z2)
gamma5 = 1j * gamma0 @ gamma1 @ gamma2 @ gamma3

C_std = 1j * gamma2 @ gamma0
T_std = 1j * gamma1 @ gamma3
P_std = gamma0
CPT_spinor = C_std @ P_std.conj() @ T_std.conj()

# AZ operators
Gamma_AZ = gamma0 @ gamma5
T_AZ = gamma0 @ gamma2
C_AZ = gamma2 @ gamma5

# ---- Walk construction ----

def proj_plus(tau):  return 0.5 * (I4 + tau)
def proj_minus(tau): return 0.5 * (I4 - tau)
def frame_transport(tau_from, tau_to):
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    return (I4 + prod) / (2 * np.sqrt((1 + cos_theta) / 2))

def build_walk_periodic(N, taus, phi):
    dim = 4 * N
    S = np.zeros((dim, dim), dtype=complex)
    for n in range(N):
        Pp, Pm = proj_plus(taus[n]), proj_minus(taus[n])
        nn, np_ = (n+1)%N, (n-1)%N
        S[4*nn:4*nn+4, 4*n:4*n+4] += frame_transport(taus[n], taus[nn]) @ Pp
        S[4*np_:4*np_+4, 4*n:4*n+4] += frame_transport(taus[n], taus[np_]) @ Pm
    V = np.eye(dim, dtype=complex)
    if phi != 0:
        cp, sp = np.cos(phi), np.sin(phi)
        for n in range(N):
            Pp, Pm = proj_plus(taus[n]), proj_minus(taus[n])
            pp_b, pm_b = np.zeros((4,2),dtype=complex), np.zeros((4,2),dtype=complex)
            np_f = nm_f = 0
            for col in range(4):
                if np_f < 2:
                    v = Pp[:,col].copy()
                    for j in range(np_f): v -= np.vdot(pp_b[:,j],v)*pp_b[:,j]
                    nm = np.real(np.vdot(v,v))
                    if nm > 1e-10: pp_b[:,np_f] = v/np.sqrt(nm); np_f += 1
                if nm_f < 2:
                    v = Pm[:,col].copy()
                    for j in range(nm_f): v -= np.vdot(pm_b[:,j],v)*pm_b[:,j]
                    nm = np.real(np.vdot(v,v))
                    if nm > 1e-10: pm_b[:,nm_f] = v/np.sqrt(nm); nm_f += 1
            M = np.zeros((4,4),dtype=complex)
            for j in range(2):
                M += np.outer(pm_b[:,j],pp_b[:,j].conj()) + np.outer(pp_b[:,j],pm_b[:,j].conj())
            V[4*n:4*n+4, 4*n:4*n+4] = cp*I4 + 1j*sp*M
    return V @ S


def apply_CPT(v, N, U_spinor):
    """Apply CPT: spinor transform U + spatial reversal n→N-1-n."""
    out = np.zeros_like(v)
    for n in range(N):
        nr = N - 1 - n
        out[4*n:4*n+4] = U_spinor @ v[4*nr:4*nr+4]
    return out


def apply_antiunitary(v, N, U_spinor):
    """Apply antiunitary U·K (spinor transform + complex conjugation, no reversal)."""
    out = np.zeros_like(v)
    for n in range(N):
        out[4*n:4*n+4] = U_spinor @ v[4*n:4*n+4].conj()
    return out


def subspace_residual(W, evecs, E, E_cut, op_fn, target_fn, N):
    """
    Measure symmetry residual restricted to |E| < E_cut subspace.

    op_fn(v) applies the symmetry operator to eigenvector v.
    target_fn(v, lam) returns what the result should be if the symmetry holds.

    Returns RMS of ||op(v) - target(v)|| over selected eigenvectors.
    """
    mask = np.abs(E) < E_cut
    if mask.sum() == 0:
        return np.nan, 0

    residuals = []
    for j in np.where(mask)[0]:
        v = evecs[:, j]
        ov = op_fn(v)
        tv = target_fn(v, np.exp(1j * E[j]))
        residuals.append(np.linalg.norm(ov - tv))

    return np.mean(residuals), mask.sum()


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 80

    print("=" * 70)
    print("Test D3: CPT symmetry in the long-wavelength limit")
    print(f"  N = {N} sites (periodic BC)")
    print("=" * 70)

    taus = build_taus(N)

    # ================================================================
    # Part 1: CPT residual vs energy cutoff at fixed N
    # ================================================================
    print("\n--- Part 1: CPT residual vs energy cutoff ---\n")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, phi in enumerate([0.0, 0.08, 0.20]):
        W = build_walk_periodic(N, taus, phi)
        evals, evecs = np.linalg.eig(W)
        E = np.angle(evals)

        # Sort
        idx = np.argsort(E)
        E = E[idx]
        evecs = evecs[:, idx]

        # Define operator functions for each symmetry
        def cpt_op(v):
            return apply_CPT(v, N, CPT_spinor)

        def chiral_op(v):
            out = np.zeros_like(v)
            for n in range(N):
                out[4*n:4*n+4] = Gamma_AZ @ v[4*n:4*n+4]
            return out

        def c_std_op(v):
            return apply_antiunitary(v, N, C_std)

        def trs_az_op(v):
            return apply_antiunitary(v, N, T_AZ)

        def phs_az_op(v):
            return apply_antiunitary(v, N, C_AZ)

        # For CPT (unitary, maps W→W): target is same eigenvector
        # CPT·W·CPT† = W means CPT maps eigenvector of W to eigenvector of W
        # at the same eigenvalue. Measure by: is CPT(v) an eigenvector of W?
        # ||W·CPT(v) - λ·CPT(v)||

        # For chiral (ΓWΓ†=W†): Γv should be eigenvector of W† with eval λ,
        # i.e., eigenvector of W with eval λ̄. So W·Γv = λ̄·Γv.

        # For TRS (TW*T†=W†): Tv* should be eigenvector of W with eval λ.

        # For PHS (CW*C†=W): Cv* should be eigenvector of W with eval λ̄.

        operators = [
            ("CPT",      cpt_op,     lambda v, lam: lam,      False),
            ("Chiral γ⁰γ⁵", chiral_op, lambda v, lam: lam.conj(), False),
            ("C_std",    c_std_op,   lambda v, lam: lam,      True),  # maps to same eval (TRS-like)
            ("T_AZ",     trs_az_op,  lambda v, lam: lam,      True),  # maps to same eval
            ("C_AZ",     phs_az_op,  lambda v, lam: lam.conj(), True),  # maps to conj eval
        ]

        E_cuts = np.linspace(0.02, np.pi, 60)

        print(f"  φ = {phi}:")

        for op_name, op_fn, target_eval_fn, is_antiunitary in operators:
            residuals = []
            counts = []
            for E_cut in E_cuts:
                mask = np.abs(E) < E_cut
                if mask.sum() == 0:
                    residuals.append(np.nan)
                    counts.append(0)
                    continue

                res_list = []
                for j in np.where(mask)[0]:
                    v = evecs[:, j]
                    ov = op_fn(v)
                    ov /= np.linalg.norm(ov)  # normalize
                    target_lam = target_eval_fn(v, np.exp(1j * E[j]))
                    # Check: is W·ov = target_lam · ov?
                    Wov = W @ ov
                    res = np.linalg.norm(Wov - target_lam * ov)
                    res_list.append(res)

                residuals.append(np.mean(res_list))
                counts.append(mask.sum())

            residuals = np.array(residuals)

            # Print key values
            for ec_val in [0.1, 0.5, 1.0, np.pi]:
                idx_c = np.argmin(np.abs(E_cuts - ec_val))
                if not np.isnan(residuals[idx_c]):
                    print(f"    {op_name:<16} |E|<{ec_val:.1f}: residual={residuals[idx_c]:.4f} "
                          f"({counts[idx_c]} states)")

        # Plot: residual vs E_cut for CPT and chiral
        ax = axes[0][col]
        for op_name, op_fn, target_eval_fn, is_anti in operators:
            residuals = []
            for E_cut in E_cuts:
                mask = np.abs(E) < E_cut
                if mask.sum() == 0:
                    residuals.append(np.nan)
                    continue
                res_list = []
                for j in np.where(mask)[0]:
                    v = evecs[:, j]
                    ov = op_fn(v)
                    ov /= np.linalg.norm(ov)
                    target_lam = target_eval_fn(v, np.exp(1j * E[j]))
                    res = np.linalg.norm(W @ ov - target_lam * ov)
                    res_list.append(res)
                residuals.append(np.mean(res_list))

            ax.plot(E_cuts, residuals, label=op_name, linewidth=1.5)

        ax.set_xlabel('Energy cutoff |E| < E_cut')
        ax.set_ylabel('Mean eigenvector residual')
        ax.set_title(f'φ = {phi}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-16)

        print()

    # ================================================================
    # Part 2: Scaling with N at fixed energy cutoff
    # ================================================================
    print("--- Part 2: CPT residual scaling with N ---\n")

    Ns = [20, 40, 60, 80, 120, 160]
    phi = 0.08
    E_cut = 0.3

    print(f"  E_cut = {E_cut}, φ = {phi}")
    print(f"  {'N':>6} {'CPT':>10} {'Chiral':>10} {'C_std':>10} {'T_AZ':>10} {'C_AZ':>10} {'#states':>8}")

    results = {name: [] for name in ["CPT", "Chiral", "C_std", "T_AZ", "C_AZ"]}
    n_states_list = []

    for Ni in Ns:
        taus_i = build_taus(Ni)
        W = build_walk_periodic(Ni, taus_i, phi)
        evals, evecs = np.linalg.eig(W)
        E = np.angle(evals)
        idx = np.argsort(E)
        E, evecs = E[idx], evecs[:, idx]

        mask = np.abs(E) < E_cut
        n_states = mask.sum()
        n_states_list.append(n_states)

        def measure_residual(op_fn, target_eval_fn):
            if n_states == 0:
                return np.nan
            res_list = []
            for j in np.where(mask)[0]:
                v = evecs[:, j]
                ov = op_fn(v)
                ov /= np.linalg.norm(ov)
                target_lam = target_eval_fn(v, np.exp(1j * E[j]))
                res = np.linalg.norm(W @ ov - target_lam * ov)
                res_list.append(res)
            return np.mean(res_list)

        def mk_cpt(Ni_):
            def f(v): return apply_CPT(v, Ni_, CPT_spinor)
            return f
        def mk_chiral(Ni_):
            def f(v):
                out = np.zeros_like(v)
                for n in range(Ni_):
                    out[4*n:4*n+4] = Gamma_AZ @ v[4*n:4*n+4]
                return out
            return f
        def mk_anti(Ni_, U):
            def f(v): return apply_antiunitary(v, Ni_, U)
            return f

        r_cpt = measure_residual(mk_cpt(Ni), lambda v,l: l)
        r_chi = measure_residual(mk_chiral(Ni), lambda v,l: l.conj())
        r_cstd = measure_residual(mk_anti(Ni, C_std), lambda v,l: l)
        r_taz = measure_residual(mk_anti(Ni, T_AZ), lambda v,l: l)
        r_caz = measure_residual(mk_anti(Ni, C_AZ), lambda v,l: l.conj())

        results["CPT"].append(r_cpt)
        results["Chiral"].append(r_chi)
        results["C_std"].append(r_cstd)
        results["T_AZ"].append(r_taz)
        results["C_AZ"].append(r_caz)

        print(f"  {Ni:>6} {r_cpt:>10.4f} {r_chi:>10.4f} {r_cstd:>10.4f} "
              f"{r_taz:>10.4f} {r_caz:>10.4f} {n_states:>8}")

    # Plot scaling
    ax = axes[1][0]
    for name in results:
        ax.plot(Ns, results[name], 'o-', label=name, markersize=4)
    ax.set_xlabel('N (chain length)')
    ax.set_ylabel('Mean residual')
    ax.set_title(f'Scaling with N (|E|<{E_cut}, φ={phi})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-16)

    # Part 2b: scaling with E_cut at fixed large N
    ax = axes[1][1]
    Ni = N
    taus_i = build_taus(Ni)
    W = build_walk_periodic(Ni, taus_i, phi)
    evals, evecs = np.linalg.eig(W)
    E = np.angle(evals)
    idx = np.argsort(E); E, evecs = E[idx], evecs[:,idx]

    E_cuts_2 = np.linspace(0.05, 1.5, 40)
    for name, op_fn_maker, target_fn in [
        ("CPT", lambda: mk_cpt(Ni), lambda v,l: l),
        ("Chiral", lambda: mk_chiral(Ni), lambda v,l: l.conj()),
        ("T_AZ", lambda: mk_anti(Ni, T_AZ), lambda v,l: l),
        ("C_AZ", lambda: mk_anti(Ni, C_AZ), lambda v,l: l.conj()),
    ]:
        op_fn = op_fn_maker()
        res = []
        for ec in E_cuts_2:
            mask = np.abs(E) < ec
            if mask.sum() == 0:
                res.append(np.nan); continue
            rl = []
            for j in np.where(mask)[0]:
                v = evecs[:,j]; ov = op_fn(v); ov /= np.linalg.norm(ov)
                tl = target_fn(v, np.exp(1j*E[j]))
                rl.append(np.linalg.norm(W @ ov - tl * ov))
            res.append(np.mean(rl))
        ax.plot(E_cuts_2, res, label=name, linewidth=1.5)

    ax.set_xlabel('Energy cutoff E_cut')
    ax.set_ylabel('Mean residual')
    ax.set_title(f'vs energy cutoff (N={Ni}, φ={phi})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-16)

    # Part 2c: check if CPT residual scales as 1/N or 1/N²
    ax = axes[1][2]
    Ns_arr = np.array(Ns, dtype=float)
    cpt_arr = np.array(results["CPT"])
    valid = ~np.isnan(cpt_arr) & (cpt_arr > 0)
    if valid.sum() >= 2:
        # Fit log(residual) = a + b*log(N)
        coeffs = np.polyfit(np.log(Ns_arr[valid]), np.log(cpt_arr[valid]), 1)
        slope = coeffs[0]
        fit_line = np.exp(np.polyval(coeffs, np.log(Ns_arr)))

        ax.loglog(Ns_arr, cpt_arr, 'ko-', label='CPT data', markersize=5)
        ax.loglog(Ns_arr, fit_line, 'r--', label=f'fit: N^{{{slope:.2f}}}')
        ax.loglog(Ns_arr, cpt_arr[0] * (Ns_arr[0]/Ns_arr), 'b:', alpha=0.5, label='1/N ref')

        # Also fit chiral
        chi_arr = np.array(results["Chiral"])
        valid_c = ~np.isnan(chi_arr) & (chi_arr > 0)
        if valid_c.sum() >= 2:
            coeffs_c = np.polyfit(np.log(Ns_arr[valid_c]), np.log(chi_arr[valid_c]), 1)
            ax.loglog(Ns_arr, chi_arr, 'gs-', label=f'Chiral: N^{{{coeffs_c[0]:.2f}}}', markersize=4)

        ax.set_xlabel('N')
        ax.set_ylabel('Mean residual (|E|<0.3)')
        ax.set_title(f'Power-law scaling (φ={phi})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Test D3: CPT in the long-wavelength limit', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = '/tmp/test_D3_CPT_continuum.png'
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved to {out}")


if __name__ == '__main__':
    main()
