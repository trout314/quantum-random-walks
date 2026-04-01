#!/usr/bin/env python3
"""
C² quantum walk on the BC helix.

Uses 2-component spinors with τ_n = d̂(n) · σ (Pauli matrices)
instead of the 4-component τ from the Dirac representation.

Compares dispersion, gauge structure, and wavepacket dynamics
against the C⁴ walk.
"""
import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.helix_geometry import exit_direction

# Pauli matrices
sigma = np.array([
    [[0, 1], [1, 0]],       # σ_x
    [[0, -1j], [1j, 0]],    # σ_y
    [[1, 0], [0, -1]],      # σ_z
], dtype=complex)


def make_tau2(d):
    """τ = d̂ · σ, a 2×2 Hermitian involution."""
    return d[0] * sigma[0] + d[1] * sigma[1] + d[2] * sigma[2]


def proj_plus2(tau):
    return 0.5 * (np.eye(2, dtype=complex) + tau)


def proj_minus2(tau):
    return 0.5 * (np.eye(2, dtype=complex) - tau)


def frame_transport2(tau_from, tau_to):
    """
    SU(2) frame transport: minimal rotation taking tau_from to tau_to.

    U = (I + tau_to @ tau_from) / (2 cos(θ/2))

    Same formula as C⁴ but with 2×2 matrices.
    """
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 2  # Tr(τ_to τ_from)/2 = d_to · d_from
    cos_half = np.sqrt(max((1 + cos_theta) / 2, 1e-15))
    return (np.eye(2, dtype=complex) + prod) / (2 * cos_half)


def build_chain_taus2(N):
    """Build 2×2 τ operators for N sites along BC helix."""
    taus = np.zeros((N, 2, 2), dtype=complex)
    for n in range(N):
        d = exit_direction(n)
        taus[n] = make_tau2(d)
    return taus


def build_coin2(tau, phi):
    """
    Coin operator in C².

    Direct analog of C⁴ build_vmix_block:
    V = cos(φ) I + i sin(φ) (|−⟩⟨+| + |+⟩⟨−|)

    where |±⟩ are eigenvectors of τ. The mixing operator M is unique
    up to a U(1) phase (the gauge freedom). We fix the phase by
    using the eigenvectors directly.
    """
    if phi == 0:
        return np.eye(2, dtype=complex)
    # Eigenvectors of τ
    evals, evecs = np.linalg.eigh(tau)
    # eigh sorts: evals[0] = -1, evals[1] = +1
    minus_vec = evecs[:, 0]  # |−⟩
    plus_vec = evecs[:, 1]   # |+⟩
    M = np.outer(minus_vec, plus_vec.conj()) + np.outer(plus_vec, minus_vec.conj())
    return np.cos(phi) * np.eye(2, dtype=complex) + 1j * np.sin(phi) * M


def build_full_walk2(N, taus, mix_phi):
    """Build the full 2N × 2N walk operator."""
    dim = 2 * N
    S = np.zeros((dim, dim), dtype=complex)

    for n in range(N):
        n_fwd = (n + 1) % N
        n_bwd = (n - 1) % N
        Pp = proj_plus2(taus[n])
        Pm = proj_minus2(taus[n])
        U_fwd = frame_transport2(taus[n], taus[n_fwd])
        U_bwd = frame_transport2(taus[n], taus[n_bwd])
        S[2*n_fwd:2*n_fwd+2, 2*n:2*n+2] += U_fwd @ Pp
        S[2*n_bwd:2*n_bwd+2, 2*n:2*n+2] += U_bwd @ Pm

    V = np.eye(dim, dtype=complex)
    if mix_phi != 0:
        for n in range(N):
            V[2*n:2*n+2, 2*n:2*n+2] = build_coin2(taus[n], mix_phi)
    return V @ S


def compute_bands2(N, taus, mix_phi, M_harm=1, n_k=400):
    """Compute quasi-Bloch band structure for C² walk."""
    THETA_BC = np.arccos(-2/3)
    ns = np.arange(N)

    # Walk blocks: (Wψ)(n) = F_n ψ(n-1) + G_n ψ(n+1)
    F_blocks = np.zeros((N, 2, 2), dtype=complex)
    G_blocks = np.zeros((N, 2, 2), dtype=complex)
    for m in range(N):
        C_m = build_coin2(taus[m], mix_phi)
        m_prev = (m - 1) % N
        m_next = (m + 1) % N
        U_fwd = frame_transport2(taus[m_prev], taus[m])
        U_bwd = frame_transport2(taus[m_next], taus[m])
        F_blocks[m] = C_m @ U_fwd @ proj_plus2(taus[m_prev])
        G_blocks[m] = C_m @ U_bwd @ proj_minus2(taus[m_next])

    # Fourier coefficients
    F_hat = np.zeros((2*M_harm+1, 2, 2), dtype=complex)
    G_hat = np.zeros((2*M_harm+1, 2, 2), dtype=complex)
    for l_idx, l in enumerate(range(-M_harm, M_harm+1)):
        exp_f = np.exp(-1j * l * ns * THETA_BC)
        for i in range(2):
            for j in range(2):
                F_hat[l_idx, i, j] = np.mean(F_blocks[:, i, j] * exp_f)
                G_hat[l_idx, i, j] = np.mean(G_blocks[:, i, j] * exp_f)

    # Sweep k
    k_vals = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    dim_wk = 2 * (2*M_harm+1)
    E_bands = np.zeros((n_k, dim_wk))

    for ik, k in enumerate(k_vals):
        Wk = np.zeros((dim_wk, dim_wk), dtype=complex)
        for mp in range(-M_harm, M_harm+1):
            for m in range(-M_harm, M_harm+1):
                l_idx = (mp - m) + M_harm
                if 0 <= l_idx < 2*M_harm+1:
                    row = (mp + M_harm) * 2
                    col = (m + M_harm) * 2
                    Wk[row:row+2, col:col+2] += (
                        np.exp(-1j*k) * F_hat[l_idx] +
                        np.exp(1j*k) * G_hat[l_idx])
        evals_k = np.linalg.eigvals(Wk)
        E_bands[ik] = np.sort(np.angle(evals_k))

    return k_vals, E_bands


def compute_c4_bands(N, phi, M_harm=1, n_k=400):
    """Compute C⁴ bands for comparison."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from quasi_bloch_l1 import (build_chain_taus, build_vmix_block,
        proj_plus, proj_minus, frame_transport as ft4)
    THETA_BC = np.arccos(-2/3)
    taus4 = build_chain_taus(N)
    ns = np.arange(N)

    F4 = np.zeros((N, 4, 4), dtype=complex)
    G4 = np.zeros((N, 4, 4), dtype=complex)
    for m in range(N):
        V_m = build_vmix_block(taus4[m], phi)
        m_prev = (m - 1) % N
        m_next = (m + 1) % N
        U_fwd = ft4(taus4[m_prev], taus4[m])
        U_bwd = ft4(taus4[m_next], taus4[m])
        F4[m] = V_m @ U_fwd @ proj_plus(taus4[m_prev])
        G4[m] = V_m @ U_bwd @ proj_minus(taus4[m_next])

    F4h = np.zeros((2*M_harm+1, 4, 4), dtype=complex)
    G4h = np.zeros((2*M_harm+1, 4, 4), dtype=complex)
    for l_idx, l in enumerate(range(-M_harm, M_harm+1)):
        exp_f = np.exp(-1j * l * ns * THETA_BC)
        for i in range(4):
            for j in range(4):
                F4h[l_idx, i, j] = np.mean(F4[:, i, j] * exp_f)
                G4h[l_idx, i, j] = np.mean(G4[:, i, j] * exp_f)

    k_vals = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    dim4 = 4 * (2*M_harm+1)
    E4 = np.zeros((n_k, dim4))
    for ik, kv in enumerate(k_vals):
        Wk = np.zeros((dim4, dim4), dtype=complex)
        for mp in range(-M_harm, M_harm+1):
            for m in range(-M_harm, M_harm+1):
                l_idx = (mp - m) + M_harm
                if 0 <= l_idx < 2*M_harm+1:
                    row = (mp + M_harm) * 4
                    col = (m + M_harm) * 4
                    Wk[row:row+4, col:col+4] += (
                        np.exp(-1j*kv) * F4h[l_idx] +
                        np.exp(1j*kv) * G4h[l_idx])
        evals_k = np.linalg.eigvals(Wk)
        E4[ik] = np.sort(np.angle(evals_k))
    return k_vals, E4


def main():
    N = 800
    print(f"Building C² walk on BC helix, N={N}")

    taus = build_chain_taus2(N)

    # Verify basic properties
    print("\n--- Verification ---")
    print(f"τ₀ = d̂(0) · σ:")
    print(taus[0])
    print(f"τ₀² = I? {np.allclose(taus[0] @ taus[0], np.eye(2))}")
    print(f"Tr(τ₀) = {np.trace(taus[0]):.6f}")

    # Check unitarity
    print("\n--- Unitarity check ---")
    for phi in [0.0, 0.10]:
        W = build_full_walk2(N, taus, phi)
        WdW = W.conj().T @ W
        err = np.max(np.abs(WdW - np.eye(2*N)))
        print(f"φ={phi:.2f}: ||W†W - I||_max = {err:.2e}")

    # Dispersion at several φ
    print("\n--- Band structure ---")
    phi_values = [0.0, 0.10, 0.20, 0.40]

    fig, axes = plt.subplots(2, len(phi_values),
                             figsize=(4*len(phi_values), 10))

    for col, phi in enumerate(phi_values):
        print(f"\nφ = {phi:.2f}:")

        # C² bands
        k, bands2 = compute_bands2(N, taus, phi, M_harm=1, n_k=400)
        idx0 = np.argmin(np.abs(k))
        b0 = np.sort(bands2[idx0])
        print(f"  C² bands at k=0: {b0}")
        print(f"  C² min |E|: {np.min(np.abs(bands2)):.6f}")
        sums = [b0[i] + b0[-(i+1)] for i in range(len(b0)//2)]
        print(f"  C² ±E sums: {[f'{s:.6f}' for s in sums]}")

        # C⁴ bands
        _, bands4 = compute_c4_bands(N, phi, M_harm=1, n_k=400)
        b4_0 = np.sort(bands4[idx0])
        phys4 = b4_0[np.abs(b4_0) < 0.8]
        print(f"  C⁴ physical bands at k=0: {phys4}")

        # Top row: C²
        ax = axes[0, col]
        for b in range(bands2.shape[1]):
            ax.plot(k, bands2[:, b], '-', lw=1.5, alpha=0.8)
        ax.set_title(f'C², φ={phi:.2f}', fontsize=13, fontweight='bold')
        ax.set_xlabel('k')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-np.pi, np.pi)
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.grid(True, alpha=0.15)

        # Bottom row: C⁴
        ax = axes[1, col]
        for b in range(bands4.shape[1]):
            ax.plot(k, bands4[:, b], '-', lw=1.5, alpha=0.8)
        ax.set_title(f'C⁴, φ={phi:.2f}', fontsize=13, fontweight='bold')
        ax.set_xlabel('k')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-np.pi, np.pi)
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.grid(True, alpha=0.15)

    axes[0, 0].set_ylabel('E (C² walk)', fontsize=12)
    axes[1, 0].set_ylabel('E (C⁴ walk)', fontsize=12)
    plt.suptitle('C² vs C⁴ quantum walk on BC helix', fontsize=15, y=1.01)
    plt.tight_layout()
    out = '/tmp/c2_vs_c4_bands.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved to {out}")


def continuum_limit():
    """
    Check the continuum limit of the C² walk two ways:

    (1) Extract effective Hamiltonian from W(k) ≈ exp(-i H_eff(k))
        at small k, and check if H_eff = σ·k + mass term (Dirac)
        or something else.

    (2) Wavepacket dynamics: launch a Gaussian and check for
        symmetric vs asymmetric L/R splitting.
    """
    N = 800
    taus = build_chain_taus2(N)

    print("=" * 60)
    print("CONTINUUM LIMIT ANALYSIS: C² walk")
    print("=" * 60)

    # ================================================================
    # (1) Effective Hamiltonian from W(k)
    # ================================================================
    print("\n--- (1) Effective Hamiltonian H_eff(k) = i log W(k) ---")

    # Use M_harm=1 (6×6 W(k)) — extract the physical 2×2 block
    # But actually, let's look at the full W(k) eigenvalues first
    for phi in [0.0, 0.10]:
        print(f"\nφ = {phi:.2f}:")
        k_vals, bands = compute_bands2(N, taus, phi, M_harm=1, n_k=800)

        # Find the physical bands (closest to E=0)
        idx0 = np.argmin(np.abs(k_vals))
        band_at_0 = np.abs(bands[idx0])
        phys_idx = np.argsort(band_at_0)[:2]  # 2 physical bands in C²

        # Extract dispersion of physical bands near k=0
        mask = np.abs(k_vals) < 0.3
        k_small = k_vals[mask]
        for bi, pidx in enumerate(sorted(phys_idx)):
            E_phys = bands[mask, pidx]
            # Fit E = a + b*k + c*k² near k=0
            coeffs = np.polyfit(k_small, E_phys, 2)
            c, b, a = coeffs
            print(f"  Physical band {bi}: E ≈ {a:.6f} + {b:.6f}·k + {c:.6f}·k²")
            if abs(b) > 0.01:
                print(f"    → group velocity v_g = {b:.4f}  (Dirac would give ±1)")

        # For Dirac: E = ±√(k² + m²) ≈ ±m ± k²/(2m), linear term = 0
        # Check if any band has |dE/dk| = 1 at small k
        print(f"  (Dirac prediction: E ≈ ±m + k²/(2m), no linear term)")

    # ================================================================
    # (2) Effective Hamiltonian matrix at each k
    # ================================================================
    print("\n--- (2) W(k) eigenvalue structure ---")

    # Build W(k) at k=0 and small k, look at eigenvalue phases
    THETA_BC = np.arccos(-2/3)
    M_harm = 1
    ns = np.arange(N)

    for phi in [0.0]:
        F_blocks = np.zeros((N, 2, 2), dtype=complex)
        G_blocks = np.zeros((N, 2, 2), dtype=complex)
        for m in range(N):
            C_m = build_coin2(taus[m], phi)
            m_prev = (m - 1) % N
            m_next = (m + 1) % N
            U_fwd = frame_transport2(taus[m_prev], taus[m])
            U_bwd = frame_transport2(taus[m_next], taus[m])
            F_blocks[m] = C_m @ U_fwd @ proj_plus2(taus[m_prev])
            G_blocks[m] = C_m @ U_bwd @ proj_minus2(taus[m_next])

        F_hat = np.zeros((2*M_harm+1, 2, 2), dtype=complex)
        G_hat = np.zeros((2*M_harm+1, 2, 2), dtype=complex)
        for l_idx, l in enumerate(range(-M_harm, M_harm+1)):
            exp_f = np.exp(-1j * l * ns * THETA_BC)
            for i in range(2):
                for j in range(2):
                    F_hat[l_idx, i, j] = np.mean(F_blocks[:, i, j] * exp_f)
                    G_hat[l_idx, i, j] = np.mean(G_blocks[:, i, j] * exp_f)

        # At k=0: examine the 6×6 W(k=0)
        dim_wk = 2 * (2*M_harm+1)
        Wk0 = np.zeros((dim_wk, dim_wk), dtype=complex)
        for mp in range(-M_harm, M_harm+1):
            for m_idx in range(-M_harm, M_harm+1):
                l_idx = (mp - m_idx) + M_harm
                if 0 <= l_idx < 2*M_harm+1:
                    row = (mp + M_harm) * 2
                    col = (m_idx + M_harm) * 2
                    Wk0[row:row+2, col:col+2] += F_hat[l_idx] + G_hat[l_idx]

        evals0 = np.linalg.eigvals(Wk0)
        phases0 = np.sort(np.angle(evals0))
        print(f"\nφ=0, W(k=0) eigenvalue phases: {phases0}")
        print(f"  ±E check: sum of outer pair = {phases0[0]+phases0[-1]:.6f}")
        print(f"  ±E check: sum of middle pair = {phases0[2]+phases0[3]:.6f}")

    # ================================================================
    # (3) Wavepacket dynamics
    # ================================================================
    print("\n--- (3) Wavepacket dynamics ---")

    sigma = 30.0
    center = N // 2
    xs = np.arange(N) - center

    for phi in [0.0, 0.10]:
        print(f"\nφ = {phi:.2f}:")
        W = build_full_walk2(N, taus, phi)

        # Initial state: Gaussian × eigenvector of τ at center
        # Use the +1 eigenvector (right-mover)
        evals, evecs = np.linalg.eigh(taus[center])
        chi_plus = evecs[:, 1]  # +1 eigenvector

        psi = np.zeros(2*N, dtype=complex)
        for n in range(N):
            env = np.exp(-0.5 * ((n - center) / sigma)**2)
            # Transport chi_plus to site n using frame transport
            if n == center:
                chi_n = chi_plus.copy()
            else:
                # Simple: just use chi_plus everywhere (not frame-transported)
                # This is an approximation but OK for seeing L/R asymmetry
                chi_n = chi_plus.copy()
            psi[2*n:2*n+2] = env * chi_n
        psi /= np.linalg.norm(psi)

        # Evolve
        t_evolve = 200
        for _ in range(t_evolve):
            psi = W @ psi
        rho = np.array([np.sum(np.abs(psi[2*n:2*n+2])**2) for n in range(N)])
        rho /= np.sum(rho)

        # Find left and right peaks
        left_mask = xs < -5
        right_mask = xs > 5
        left_weight = np.sum(rho[left_mask])
        right_weight = np.sum(rho[right_mask])

        if left_mask.any() and np.max(rho[left_mask]) > 1e-6:
            left_peak = xs[left_mask][np.argmax(rho[left_mask])]
            left_vel = abs(left_peak) / t_evolve
        else:
            left_peak = 0; left_vel = 0
        if right_mask.any() and np.max(rho[right_mask]) > 1e-6:
            right_peak = xs[right_mask][np.argmax(rho[right_mask])]
            right_vel = abs(right_peak) / t_evolve
        else:
            right_peak = 0; right_vel = 0

        print(f"  After t={t_evolve}:")
        print(f"    Left peak at x={left_peak}, v_L={left_vel:.4f}, weight={left_weight:.4f}")
        print(f"    Right peak at x={right_peak}, v_R={right_vel:.4f}, weight={right_weight:.4f}")
        print(f"    L/R weight ratio: {left_weight/max(right_weight,1e-10):.4f}")
        print(f"    (Dirac predicts v_L = v_R and equal weights)")

    # ================================================================
    # (4) Same test for C⁴ walk as control
    # ================================================================
    print("\n--- (4) C⁴ walk wavepacket (control) ---")
    from quasi_bloch_l1 import (build_chain_taus as bct4, build_full_walk as bfw4,
        proj_plus as pp4)
    taus4 = bct4(N)

    for phi in [0.0, 0.10]:
        print(f"\nφ = {phi:.2f}:")
        W4 = bfw4(N, taus4, phi)

        Pp0 = pp4(taus4[center])
        chi4 = Pp0[:, 0]
        chi4 /= np.linalg.norm(chi4)

        psi4 = np.zeros(4*N, dtype=complex)
        for n in range(N):
            env = np.exp(-0.5 * ((n - center) / sigma)**2)
            psi4[4*n:4*n+4] = env * chi4
        psi4 /= np.linalg.norm(psi4)

        t_evolve = 200
        for _ in range(t_evolve):
            psi4 = W4 @ psi4
        rho4 = np.array([np.sum(np.abs(psi4[4*n:4*n+4])**2) for n in range(N)])
        rho4 /= np.sum(rho4)

        left_mask = xs < -5
        right_mask = xs > 5
        left_w4 = np.sum(rho4[left_mask])
        right_w4 = np.sum(rho4[right_mask])

        if left_mask.any() and np.max(rho4[left_mask]) > 1e-6:
            lp4 = xs[left_mask][np.argmax(rho4[left_mask])]
            lv4 = abs(lp4) / t_evolve
        else:
            lp4 = 0; lv4 = 0
        if right_mask.any() and np.max(rho4[right_mask]) > 1e-6:
            rp4 = xs[right_mask][np.argmax(rho4[right_mask])]
            rv4 = abs(rp4) / t_evolve
        else:
            rp4 = 0; rv4 = 0

        print(f"  After t={t_evolve}:")
        print(f"    Left peak at x={lp4}, v_L={lv4:.4f}, weight={left_w4:.4f}")
        print(f"    Right peak at x={rp4}, v_R={rv4:.4f}, weight={right_w4:.4f}")
        print(f"    L/R weight ratio: {left_w4/max(right_w4,1e-10):.4f}")

    # ================================================================
    # Plot comparison
    # ================================================================
    print("\n--- Generating wavepacket comparison plot ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, phi in enumerate([0.0, 0.10]):
        # C² walk
        W2 = build_full_walk2(N, taus, phi)
        evals_c, evecs_c = np.linalg.eigh(taus[center])
        chi2 = evecs_c[:, 1]
        psi2 = np.zeros(2*N, dtype=complex)
        for n in range(N):
            env = np.exp(-0.5 * ((n - center) / sigma)**2)
            psi2[2*n:2*n+2] = env * chi2
        psi2 /= np.linalg.norm(psi2)
        for _ in range(200):
            psi2 = W2 @ psi2
        rho2 = np.array([np.sum(np.abs(psi2[2*n:2*n+2])**2) for n in range(N)])

        # C⁴ walk
        W4 = bfw4(N, taus4, phi)
        chi4 = pp4(taus4[center])[:, 0]
        chi4 /= np.linalg.norm(chi4)
        psi4 = np.zeros(4*N, dtype=complex)
        for n in range(N):
            env = np.exp(-0.5 * ((n - center) / sigma)**2)
            psi4[4*n:4*n+4] = env * chi4
        psi4 /= np.linalg.norm(psi4)
        for _ in range(200):
            psi4 = W4 @ psi4
        rho4 = np.array([np.sum(np.abs(psi4[4*n:4*n+4])**2) for n in range(N)])

        ax = axes[0, col]
        ax.plot(xs, rho2/np.sum(rho2), 'b-', lw=1.5, label='C² walk')
        ax.set_title(f'C² walk, φ={phi:.2f}, t=200', fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('density')
        ax.set_xlim(-250, 250)
        ax.legend()
        ax.grid(True, alpha=0.15)

        ax = axes[1, col]
        ax.plot(xs, rho4/np.sum(rho4), 'r-', lw=1.5, label='C⁴ walk')
        ax.set_title(f'C⁴ walk, φ={phi:.2f}, t=200', fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('density')
        ax.set_xlim(-250, 250)
        ax.legend()
        ax.grid(True, alpha=0.15)

    plt.suptitle('Wavepacket dynamics: C² vs C⁴', fontsize=15, y=1.01)
    plt.tight_layout()
    out = '/tmp/c2_vs_c4_wavepacket.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
    continuum_limit()
