#!/usr/bin/env python3
"""
Test D: Discrete symmetries C, P, T of the 1D quantum walk.

Comprehensive analysis of charge conjugation (C), parity (P), and time
reversal (T) for the BC helix walk operator W = V · S.

Main results:
  At φ=0 (pure shift): AZ symmetry class CII
    - Chiral  Γ = γ⁰γ⁵  (anticommutes with every τ_n)
    - TRS     T = γ⁰γ²·K  (T² = -1, Kramers pairs)
    - PHS     C = γ²γ⁵·K  (C² = -1)

  At φ≠0 (with V-mixing mass term):
    - All three broken as exact walk symmetries (ordering issue)
    - Factor-wise: γ⁰γ⁵ maps S→S† and V→V† individually
    - ±E spectral pairing protected by factor-wise chiral structure
    - Kramers degeneracy broken (eigenvalues non-degenerate)
    - (-1)^n staggering still gives E↔E+π

Usage: python3 scripts/test_D_CPT_symmetries.py [N]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helix_geometry import build_taus

# ---- Dirac algebra ----
I4 = np.eye(4, dtype=complex)
I2 = np.eye(2, dtype=complex)
sigma = [
    np.array([[0,1],[1,0]], dtype=complex),
    np.array([[0,-1j],[1j,0]], dtype=complex),
    np.array([[1,0],[0,-1]], dtype=complex),
]

def blk(a, b, c, d):
    return np.block([[a,b],[c,d]])

Z2 = np.zeros((2,2), dtype=complex)
gamma0 = blk(I2, Z2, Z2, -I2)
gamma1 = blk(Z2, sigma[0], -sigma[0], Z2)
gamma2 = blk(Z2, sigma[1], -sigma[1], Z2)
gamma3 = blk(Z2, sigma[2], -sigma[2], Z2)
gamma5 = 1j * gamma0 @ gamma1 @ gamma2 @ gamma3

# Symmetry operators
Gamma_chiral = gamma0 @ gamma5   # chiral (sublattice)
T_trs = gamma0 @ gamma2          # time reversal (unitary part)
C_phs = gamma2 @ gamma5          # particle-hole (unitary part)

# ---- Walk operator construction ----

def proj_plus(tau):
    return 0.5 * (I4 + tau)

def proj_minus(tau):
    return 0.5 * (I4 - tau)

def frame_transport(tau_from, tau_to):
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    cos_half = np.sqrt((1 + cos_theta) / 2)
    return (np.eye(4) + prod) / (2 * cos_half)


def build_walk_periodic(N, taus, phi):
    """Build periodic walk operator W = V·S and return W, S, V."""
    dim = 4 * N

    # Shift
    S = np.zeros((dim, dim), dtype=complex)
    for n in range(N):
        Pp = proj_plus(taus[n])
        Pm = proj_minus(taus[n])
        nn = (n+1) % N
        np_ = (n-1) % N
        S[4*nn:4*nn+4, 4*n:4*n+4] += frame_transport(taus[n], taus[nn]) @ Pp
        S[4*np_:4*np_+4, 4*n:4*n+4] += frame_transport(taus[n], taus[np_]) @ Pm

    # V-mixing
    V = np.eye(dim, dtype=complex)
    if phi != 0:
        cp, sp = np.cos(phi), np.sin(phi)
        for n in range(N):
            Pp = proj_plus(taus[n])
            Pm = proj_minus(taus[n])
            # Gram-Schmidt for P± bases
            pp_b = np.zeros((4,2), dtype=complex)
            pm_b = np.zeros((4,2), dtype=complex)
            np_f = nm_f = 0
            for col in range(4):
                if np_f < 2:
                    v = Pp[:, col].copy()
                    for j in range(np_f):
                        v -= np.vdot(pp_b[:,j], v) * pp_b[:,j]
                    nm = np.real(np.vdot(v,v))
                    if nm > 1e-10:
                        pp_b[:, np_f] = v / np.sqrt(nm)
                        np_f += 1
                if nm_f < 2:
                    v = Pm[:, col].copy()
                    for j in range(nm_f):
                        v -= np.vdot(pm_b[:,j], v) * pm_b[:,j]
                    nm = np.real(np.vdot(v,v))
                    if nm > 1e-10:
                        pm_b[:, nm_f] = v / np.sqrt(nm)
                        nm_f += 1
            M = np.zeros((4,4), dtype=complex)
            for j in range(2):
                M += np.outer(pm_b[:,j], pp_b[:,j].conj())
                M += np.outer(pp_b[:,j], pm_b[:,j].conj())
            V[4*n:4*n+4, 4*n:4*n+4] = cp * I4 + 1j * sp * M

    return V @ S, S, V


def apply_U(U, M, N, conj=False):
    """Apply site-diagonal (U⊗I_N) M (U†⊗I_N), optionally on M*."""
    dim = 4*N
    Mw = M.conj() if conj else M
    R = np.zeros((dim, dim), dtype=complex)
    Ud = U.conj().T
    for n in range(N):
        for m in range(N):
            R[4*n:4*n+4, 4*m:4*m+4] = U @ Mw[4*n:4*n+4, 4*m:4*m+4] @ Ud
    return R


def err(A, B, N):
    return np.linalg.norm(A - B) / np.sqrt(4*N)


def count_distinct(W, tol=1e-8):
    """Count distinct eigenvalues on the unit circle."""
    E = np.sort(np.angle(np.linalg.eigvals(W)))
    n = 1
    for i in range(1, len(E)):
        if abs(E[i] - E[i-1]) > tol:
            n += 1
    return n


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    dim = 4 * N
    taus = build_taus(N)
    phis = [0.0, 0.04, 0.08, 0.12, 0.20]

    print("=" * 72)
    print("Test D: Discrete CPT symmetries of the 1D quantum walk")
    print(f"  Chain: N={N} sites (periodic BC), dim={dim}")
    print("=" * 72)

    # ---- Part 1: Exact symmetries at φ=0 ----
    print("\n--- Part 1: Symmetries at φ=0 (AZ classification) ---\n")
    W0, S0, V0 = build_walk_periodic(N, taus, 0.0)
    e_u = np.linalg.norm(W0 @ W0.conj().T - np.eye(dim)) / np.sqrt(dim)
    print(f"  Unitarity: ||WW†-I||/√dim = {e_u:.2e}")

    # Chiral
    e_ch = err(apply_U(Gamma_chiral, W0, N), W0.conj().T, N)
    print(f"\n  Chiral Γ = γ⁰γ⁵:   ||ΓWΓ† - W†|| = {e_ch:.2e}  {'✓' if e_ch < 1e-10 else '✗'}")

    # TRS
    e_trs = err(apply_U(T_trs, W0, N, conj=True), W0.conj().T, N)
    T2 = T_trs @ T_trs.conj()  # T² for antiunitary
    t2_val = int(np.round(np.real(np.trace(T2) / 4)))
    print(f"  TRS T = γ⁰γ²·K:   ||TW*T† - W†|| = {e_trs:.2e}  {'✓' if e_trs < 1e-10 else '✗'}  T²={t2_val:+d}")

    # PHS
    e_phs = err(apply_U(C_phs, W0, N, conj=True), W0, N)
    C2 = C_phs @ C_phs.conj()
    c2_val = int(np.round(np.real(np.trace(C2) / 4)))
    print(f"  PHS C = γ²γ⁵·K:   ||CW*C† - W || = {e_phs:.2e}  {'✓' if e_phs < 1e-10 else '✗'}  C²={c2_val:+d}")

    print(f"\n  → AZ class: CII (T²=-1, C²=-1, chiral)")
    print(f"  → 1D topological invariant: 2ℤ")

    # Kramers check
    nd = count_distinct(W0)
    print(f"  → Distinct eigenvalues: {nd}/{dim} (ratio {dim/nd:.1f}, Kramers {'✓' if nd == dim//2 else '✗'})")

    # ---- Part 2: Factor-by-factor at φ≠0 ----
    print("\n--- Part 2: Factor-wise action at φ≠0 ---\n")
    print(f"  γ⁰γ⁵ anticommutes with every τ_n (proven algebraically).")
    print(f"  This guarantees: γ⁰γ⁵ S (γ⁰γ⁵)† = S† and γ⁰γ⁵ V (γ⁰γ⁵)† = V†.\n")
    print(f"  But W = VS, so γ⁰γ⁵ W (γ⁰γ⁵)† = V†S† ≠ S†V† = W†.")
    print(f"  The ordering issue means Γ is not an exact walk symmetry.\n")
    print(f"  However, since V†S† = (SV)† has eigenvalues = conj(eigvals(W)),")
    print(f"  the spectral ±E pairing is PROTECTED.\n")

    print(f"  {'φ':>6} {'||ΓSΓ†-S†||':>14} {'||ΓVΓ†-V†||':>14} {'||ΓWΓ†-W†||':>14}"
          f" {'±E err':>12} {'n_distinct':>12} {'Kramers':>8}")
    print(f"  {'-'*80}")

    for phi in phis:
        W, S, V = build_walk_periodic(N, taus, phi)
        e_s = err(apply_U(Gamma_chiral, S, N), S.conj().T, N)
        e_v = err(apply_U(Gamma_chiral, V, N), V.conj().T, N)
        e_w = err(apply_U(Gamma_chiral, W, N), W.conj().T, N)

        E = np.angle(np.linalg.eigvals(W))
        Ep = np.sort(E[E >= -1e-14])
        En = np.sort(-E[E < -1e-14])
        pm = np.max(np.abs(Ep - En)) if len(Ep) == len(En) else -1

        nd = count_distinct(W)
        kr = "yes" if nd == dim // 2 else "no"
        print(f"  {phi:>6.3f} {e_s:>14.2e} {e_v:>14.2e} {e_w:>14.2e}"
              f" {pm:>12.2e} {nd:>12} {kr:>8}")

    # ---- Part 3: Staggering ----
    print(f"\n--- Part 3: Staggered sublattice symmetry ---\n")
    print(f"  (-1)^n is an exact anti-symmetry: (-1)^n W (-1)^n = -W")
    print(f"  This gives E ↔ E+π pairing (not ±E).\n")

    stag = np.diag([(-1)**n for n in range(N) for _ in range(4)])
    for phi in phis:
        W, _, _ = build_walk_periodic(N, taus, phi)
        e_stag = err(stag @ W @ stag, -W, N)
        print(f"  φ={phi:.3f}: ||(-1)^n W (-1)^n + W|| = {e_stag:.2e}")

    # ---- Part 4: Summary table ----
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print("""
  Symmetry operators (site-independent, Dirac representation):

    Operator    Matrix      Type         Factor action
    ──────────  ──────────  ───────────  ─────────────────────────
    Chiral Γ    γ⁰γ⁵       unitary      Γ S Γ† = S†, Γ V Γ† = V†
    TRS    T    γ⁰γ² · K   antiunitary   T S* T† = S†, T V* T† = V
    PHS    C    γ²γ⁵ · K   antiunitary   C S* C† = S,  C V* C† = V†
    Stagger     (-1)^n      unitary      flips sign of hopping

  Key identities:
    • {γ⁰γ⁵, τ_n} = 0  for ALL exit directions d_n
    • Γ = i · T · C†     (chiral = TRS × PHS†, up to phase)
    • T² = -1, C² = -1  → AZ class CII at φ=0

  φ = 0 (pure shift, no mass):
    • All three symmetries exact: ΓWΓ† = W†, TW*T† = W†, CW*C† = W
    • ±E spectral pairing + Kramers 2-fold degeneracy
    • Each eigenvalue appears twice; each |E| appears 4 times
    • AZ class CII, topological invariant 2ℤ in 1D

  φ ≠ 0 (V-mixing mass term):
    • Exact walk symmetries broken (V†S† ≠ S†V†)
    • ±E spectral pairing PROTECTED by factor-wise chiral structure
    • Kramers degeneracy BROKEN (eigenvalues non-degenerate)
    • (-1)^n staggering persists (E ↔ E+π)

  Physical interpretation:
    • γ⁰γ⁵ is the massless Dirac chiral operator
    • V-mixing φ is the discrete mass parameter
    • Mass breaks exact chiral/TRS/PHS symmetries
    • But the split-step factored structure W=VS protects spectral ±E
    • This is the discrete analog of: mass breaks chiral symmetry,
      but particle-antiparticle pairing persists in the spectrum
    """)

    # ---- Part 5: Visualization ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, phi in enumerate([0.0, 0.08, 0.20]):
        W, S, V = build_walk_periodic(N, taus, phi)
        evals = np.linalg.eigvals(W)
        E = np.angle(evals)

        # Top row: eigenvalues on unit circle
        ax = axes[0][idx]
        ax.scatter(np.real(evals), np.imag(evals), s=8, alpha=0.6)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')
        nd = count_distinct(W)
        ax.set_title(f'φ={phi}: {nd} distinct eigenvalues')
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        ax.grid(True, alpha=0.2)

        # Bottom row: quasienergy histogram showing ±E and E+π symmetries
        ax = axes[1][idx]
        ax.hist(E, bins=80, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(0, color='r', linewidth=1, linestyle='--', label='E=0')
        ax.axvline(np.pi, color='g', linewidth=1, linestyle='--', alpha=0.5, label='E=π')
        ax.axvline(-np.pi, color='g', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel('Quasienergy E')
        ax.set_ylabel('Count')
        ax.set_title(f'φ={phi}: ±E pairing, E↔E+π staggering')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'Test D: CPT symmetries of 1D quantum walk (N={N}, periodic)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = '/tmp/test_D_CPT_symmetries.png'
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")


if __name__ == '__main__':
    main()
