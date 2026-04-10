#!/usr/bin/env python3
"""
Test D2: Combined CPT symmetries (CP, CT, PT, CPT).

Standard Dirac-representation operators:
  C = iγ²γ⁰ · K  (charge conjugation, antiunitary)
  P = γ⁰ ⊗ R     (parity: β spinor transform + spatial reversal)
  T = iγ¹γ³ · K  (time reversal, antiunitary)

We also compare with the AZ operators found in Test D:
  Γ_AZ = γ⁰γ⁵    (chiral, unitary)
  T_AZ = γ⁰γ²·K  (TRS, antiunitary)
  C_AZ = γ²γ⁵·K  (PHS, antiunitary)

For each candidate, test whether it maps W to W, W†, -W, or -W†.
"""
import numpy as np
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

def blk(a,b,c,d):
    return np.block([[a,b],[c,d]])

gamma0 = blk(I2, Z2, Z2, -I2)
gamma1 = blk(Z2, sigma[0], -sigma[0], Z2)
gamma2 = blk(Z2, sigma[1], -sigma[1], Z2)
gamma3 = blk(Z2, sigma[2], -sigma[2], Z2)
gamma5 = 1j * gamma0 @ gamma1 @ gamma2 @ gamma3

# Standard Dirac operators (unitary parts)
C_std = 1j * gamma2 @ gamma0      # charge conjugation: iγ²γ⁰ = iα₂
T_std = 1j * gamma1 @ gamma3      # time reversal: iγ¹γ³
P_std = gamma0                     # parity: γ⁰ (+ spatial reversal)

# AZ operators from Test D
Gamma_AZ = gamma0 @ gamma5        # chiral
T_AZ = gamma0 @ gamma2            # TRS
C_AZ = gamma2 @ gamma5            # PHS

# ---- Walk construction ----

def proj_plus(tau):
    return 0.5 * (I4 + tau)

def proj_minus(tau):
    return 0.5 * (I4 - tau)

def frame_transport(tau_from, tau_to):
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    cos_half = np.sqrt((1 + cos_theta) / 2)
    return (I4 + prod) / (2 * cos_half)


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

    return V @ S


# ---- Symmetry application ----

def apply_op(U_spinor, W, N, conjugate=False, reverse=False, stagger=False):
    """
    Apply a symmetry operator to W.

    The full operator is: Σ = (phases) ⊗ U_spinor ⊗ (spatial)

    Result: Σ W Σ† (unitary) or Σ W* Σ† (antiunitary, if conjugate=True)

    Parameters:
      U_spinor: 4×4 unitary matrix (spinor transformation)
      conjugate: if True, apply to W* (for antiunitary operators)
      reverse: if True, also reverse spatial indices n → N-1-n
      stagger: if True, multiply by (-1)^n at each site
    """
    dim = 4 * N
    Ud = U_spinor.conj().T
    Wwork = W.conj() if conjugate else W.copy()

    result = np.zeros((dim, dim), dtype=complex)
    for n in range(N):
        n_out = (N-1-n) if reverse else n
        s_n = ((-1)**n) if stagger else 1
        for m in range(N):
            m_out = (N-1-m) if reverse else m
            s_m = ((-1)**m) if stagger else 1
            block = Wwork[4*n_out:4*n_out+4, 4*m_out:4*m_out+4]
            result[4*n:4*n+4, 4*m:4*m+4] = (s_n * s_m) * (U_spinor @ block @ Ud)
    return result


def err(A, B, N):
    return np.linalg.norm(A - B) / np.sqrt(4*N)


def test_operator(name, U, W, N, conjugate=False, reverse=False, stagger=False):
    """Test what an operator maps W to."""
    OWO = apply_op(U, W, N, conjugate=conjugate, reverse=reverse, stagger=stagger)
    Wd = W.conj().T

    targets = {
        ' W':   W,
        ' W†': Wd,
        '-W':  -W,
        '-W†': -Wd,
    }

    best_name = None
    best_err = 1e10
    errs = {}
    for tname, target in targets.items():
        e = err(OWO, target, N)
        errs[tname] = e
        if e < best_err:
            best_err = e
            best_name = tname

    return best_name, best_err, errs


def squaring(U, conjugate=False):
    """Compute Σ² for unitary or antiunitary operator."""
    if conjugate:
        # Antiunitary: (U·K)² = U · U*
        S2 = U @ U.conj()
    else:
        S2 = U @ U
    tr = np.real(np.trace(S2))
    if abs(tr - 4) < 0.01:
        return "+1"
    elif abs(tr + 4) < 0.01:
        return "-1"
    else:
        return f"tr={tr:.2f}"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    taus = build_taus(N)

    print("=" * 75)
    print("Test D2: Combined CPT symmetries")
    print(f"  Chain: N={N} sites (periodic BC)")
    print("=" * 75)

    # First, verify the relationship between standard and AZ operators
    print("\n--- Operator relationships ---\n")

    # Products of standard operators
    CT_std = C_std @ T_std.conj()  # CT (product of two antiunitaries → unitary)
    # Note: for antiunitaries A=U₁K and B=U₂K, AB = U₁·U₂*
    # So C·T = (iγ²γ⁰·K)(iγ¹γ³·K) = iγ²γ⁰ · (iγ¹γ³)* · K² = iγ²γ⁰ · (iγ¹γ³)*
    CT_product = C_std @ T_std.conj()  # unitary part: iγ²γ⁰ · (iγ¹γ³)*

    print("  Standard Dirac operators (unitary parts):")
    print(f"    C_std = iγ²γ⁰ = iα₂")
    print(f"    T_std = iγ¹γ³")
    print(f"    P_std = γ⁰ = β (+ spatial reversal)")
    print()

    # Check: is CT_std related to γ⁰γ⁵?
    for phase_name, phase in [('', 1), ('-', -1), ('i', 1j), ('-i', -1j)]:
        e = np.linalg.norm(CT_product - phase * Gamma_AZ)
        if e < 1e-10:
            print(f"    C_std · T_std = {phase_name}γ⁰γ⁵ (= {phase_name}Γ_AZ)  ✓")

    # Check: is C_std related to C_AZ?
    for phase_name, phase in [('', 1), ('-', -1), ('i', 1j), ('-i', -1j)]:
        e = np.linalg.norm(C_std - phase * C_AZ)
        if e < 1e-10:
            print(f"    C_std = {phase_name}C_AZ  ✓")

    # Check: is T_std related to T_AZ?
    for phase_name, phase in [('', 1), ('-', -1), ('i', 1j), ('-i', -1j)]:
        e = np.linalg.norm(T_std - phase * T_AZ)
        if e < 1e-10:
            print(f"    T_std = {phase_name}T_AZ  ✓")

    # Compute all products
    # For antiunitaries A=U_A·K, B=U_B·K: product AB = U_A · U_B*  (linear!)
    # For linear U times antiunitary A=U_A·K: UA = U·U_A · K (antiunitary)
    # For antiunitary A=U_A·K times linear U: AU = U_A · U* · K (antiunitary)

    # CP = C·P: C=(C_std·K) is antiunitary, P=(P_std⊗R) is unitary
    # CP = C_std · (P_std)* · K ⊗ R  (antiunitary)
    CP_spinor = C_std @ P_std.conj()

    # PT = P·T: P=(P_std⊗R) unitary, T=(T_std·K) antiunitary
    # PT = P_std · T_std · K ⊗ R  (antiunitary)
    PT_spinor = P_std @ T_std

    # CT = C·T: both antiunitary → unitary
    # CT = C_std · T_std* (no K, no R)
    CT_spinor = C_std @ T_std.conj()

    # CPT = C·P·T: antiunitary·unitary·antiunitary = unitary
    # CPT = C_std · P_std* · T_std*  (but with R from P)
    # More carefully: CPT = (C_std·K)(P_std⊗R)(T_std·K)
    # = C_std · K · P_std · R · T_std · K
    # = C_std · (P_std)* · K · R · T_std · K  [K moves past P_std]
    # = C_std · (P_std)* · (R · T_std)* · K²  [K moves past R·T_std, both real/spatial]
    # Wait, R is spatial, K is on spinor. They commute.
    # = C_std · (P_std·T_std)* · R  [two K's cancel, one R remains]
    # Actually: (C·K)(P⊗R)(T·K) = C(KP)⊗R · T·K = C·P*·T_std⊗R·K·K = C·P*·T*⊗R
    # Hmm let me be more careful.
    #
    # C̃ = C_std · K (acts on spinor), P̃ = P_std ⊗ R, T̃ = T_std · K
    # C̃P̃T̃ = C_std·K · P_std⊗R · T_std·K
    #       = C_std · (K·P_std) · R · (T_std·K)
    #       = C_std · P_std* · R · T_std · K · K   [K commutes with R (spatial)]
    #       Wait, K·(T_std·K) = K·T_std·K = T_std*·K² = T_std*
    #       Hmm no: K is complex conjugation on spinor components.
    #       K·P_std·R = P_std*·R·K? No, K acts on everything to its right.
    #
    # Let me think of it differently. An antiunitary A acts as A(v) = U·v*.
    # So C̃(v) = C_std · v*
    # P̃(v)(n) = P_std · v(N-1-n)
    # T̃(v) = T_std · v*
    #
    # C̃P̃T̃(v) = C̃(P̃(T̃(v)))
    # T̃(v) = T_std · v*
    # P̃(T̃(v))(n) = P_std · (T_std · v*(N-1-n)) = P_std · T_std · v*(N-1-n)
    # C̃(P̃(T̃(v)))(n) = C_std · [P_std · T_std · v*(N-1-n)]*
    #                   = C_std · P_std* · T_std* · v(N-1-n)
    #
    # So CPT is LINEAR (no conjugation), with spatial reversal R, and
    # spinor matrix = C_std · P_std* · T_std*

    CPT_spinor = C_std @ P_std.conj() @ T_std.conj()

    # Define all operators to test
    operators = [
        # (name, spinor_matrix, conjugate?, reverse?, stagger?, squaring_is_antiunitary?)
        # --- Individual ---
        ("C (iγ²γ⁰·K)",           C_std,     True,  False, False),
        ("P (γ⁰ ⊗ R)",            P_std,     False, True,  False),
        ("T (iγ¹γ³·K)",           T_std,     True,  False, False),
        # --- AZ operators (for comparison) ---
        ("Γ_AZ (γ⁰γ⁵)",          Gamma_AZ,  False, False, False),
        ("T_AZ (γ⁰γ²·K)",        T_AZ,      True,  False, False),
        ("C_AZ (γ²γ⁵·K)",        C_AZ,      True,  False, False),
        # --- Pairwise ---
        ("CP (iγ²·K ⊗ R)",        CP_spinor, True,  True,  False),
        ("CT (C·P*·unitary)",      CT_spinor, False, False, False),
        ("PT (P·T·K ⊗ R)",        PT_spinor, True,  True,  False),
        # --- Triple ---
        ("CPT (linear ⊗ R)",       CPT_spinor, False, True, False),
        # --- With staggering ---
        ("(-1)^n I",               I4,        False, False, True),
        ("(-1)^n C",               C_std,     True,  False, True),
        ("(-1)^n T",               T_std,     True,  False, True),
        ("(-1)^n P ⊗ R",          P_std,     False, True,  True),
        ("(-1)^n CT",              CT_spinor, False, False, True),
        ("(-1)^n CP ⊗ R",         CP_spinor, True,  True,  True),
        ("(-1)^n PT ⊗ R",         PT_spinor, True,  True,  True),
        ("(-1)^n CPT ⊗ R",        CPT_spinor, False, True, True),
    ]

    for phi in [0.0, 0.08, 0.20]:
        print(f"\n{'='*75}")
        print(f"φ = {phi}")
        print(f"{'='*75}")

        W = build_walk_periodic(N, taus, phi)

        print(f"\n  {'Operator':<24} {'→W':>8} {'→W†':>8} {'→-W':>8} {'→-W†':>8}  {'best':>5} {'Σ²':>5}")
        print(f"  {'-'*70}")

        for (name, U, conj, rev, stag) in operators:
            best_name, best_err, errs = test_operator(
                name, U, W, N, conjugate=conj, reverse=rev, stagger=stag)

            sq = squaring(U, conjugate=conj)

            marker = "✓" if best_err < 1e-10 else ""
            print(f"  {name:<24} "
                  f"{errs[' W']:>8.1e} {errs[' W†']:>8.1e} "
                  f"{errs['-W']:>8.1e} {errs['-W†']:>8.1e}  "
                  f"{best_name:>5} {sq:>5} {marker}")

    # ---- Summary ----
    print(f"\n\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print("""
  OPERATOR ALGEBRA:
    C_std = iγ²γ⁰ = -i · T_AZ     (standard C is AZ TRS up to phase)
    CT_std = C_std · T_std* = -iγ⁵  (NOT γ⁰γ⁵; differs by factor of γ⁰)
    Γ_AZ = T_AZ · C_AZ*             (AZ chiral = product of AZ antiunitaries)

  The walk's symmetry operators (AZ) differ from standard Dirac C/P/T
  because τ = (√7/4)β + (3/4)(d·α) mixes β and α, changing which
  operators (anti)commute with τ.

  ─────────────────────────────────────────────────────────────────────
  RESULTS:
  ─────────────────────────────────────────────────────────────────────

  C (iγ²γ⁰·K):  EXACT symmetry at φ=0 (maps W→W†).
                  Equivalent to AZ TRS (γ⁰γ²·K) up to phase -i.
                  Broken at φ≠0 by V-mixing (factor-ordering issue).

  P (γ⁰ ⊗ R):   BROKEN at all φ. The BC helix has definite handedness;
                  spatial reversal maps R-helix to L-helix, not to itself.
                  Parity is an inter-walk symmetry (R ↔ L), not intra-walk.

  T (iγ¹γ³·K):  NOT a symmetry at any φ. The standard Dirac time-reversal
                  does not (anti)commute correctly with our τ operators.
                  The walk has a DIFFERENT TRS: γ⁰γ²·K (from AZ analysis).

  CP:   Broken (inherits P breaking).
  CT:   CT_std = -iγ⁵ is NOT the chiral operator γ⁰γ⁵. Not a symmetry.
  PT:   Broken (inherits P breaking and T not being a symmetry).
  CPT:  Broken. Error ~0.5 at φ=0 (NOT protected by CPT theorem since
        this is a discrete lattice system, not a continuum QFT).

  (-1)^n: EXACT at all φ. The only universal symmetry.
    """)


if __name__ == '__main__':
    main()
