#!/usr/bin/env python3
"""
Perpendicular-space Fourier analysis of the walk operator.

The τ operator at site n depends quasiperiodically on φ(n) = nθ mod 2π,
plus the sublattice index r = n mod 4. We Fourier-analyze τ as a function
of φ to determine:

1. How rapidly the Fourier coefficients decay — this tells us how many
   harmonics are needed for a quasi-Bloch truncation
2. The structure of the coupling in (k, m) space where m is the perp-space
   harmonic index

If τ(n) = Σ_l T_l^{(r)} e^{ilnθ}, then in the Bloch basis
ψ(n) = e^{ikn} Σ_m c_m e^{imnθ}, the walk operator couples c_m to c_{m+l}
with coefficient depending on T_l. Rapid decay of T_l means few harmonics
suffice and the effective Hamiltonian is a small matrix.

Also sweeps φ_mix to show how quasi-Bloch quality changes with mass.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.helix_geometry import build_taus

def proj_plus(tau):
    return 0.5 * (np.eye(4) + tau)

def proj_minus(tau):
    return 0.5 * (np.eye(4) - tau)

def frame_transport(tau_from, tau_to):
    prod = tau_to @ tau_from
    cos_theta = np.real(np.trace(prod)) / 4
    cos_half = np.sqrt(max((1 + cos_theta) / 2, 1e-15))
    scale = 1 / (2 * cos_half)
    return scale * (np.eye(4) + prod)


THETA_BC = np.arccos(-2/3)


# ---- Fourier analysis of τ operators ----

def fourier_analyze_tau(taus, N, n_sublattices=4):
    """
    Compute Fourier coefficients of τ(n) in each sublattice.

    For sublattice r, we have sites n = r, r+4, r+8, ...
    At these sites, φ(n) = nθ mod 2π.

    We compute T_l^(r) = (1/N_r) Σ_{n in sublattice r} τ(n) e^{-ilnθ}

    Returns dict mapping r -> array of (n_harmonics, 4, 4) Fourier coefficients.
    """
    results = {}
    for r in range(n_sublattices):
        sites = np.arange(r, N, n_sublattices)
        N_r = len(sites)
        phi_r = sites * THETA_BC  # unwrapped phases

        # Compute Fourier coefficients for each (i,j) element of the 4x4 matrix
        # Use harmonics l = 0, 1, 2, ..., N_r//2
        n_harmonics = N_r // 2
        T_l = np.zeros((n_harmonics, 4, 4), dtype=complex)

        for l in range(n_harmonics):
            phases = np.exp(-1j * l * phi_r)
            for i in range(4):
                for j in range(4):
                    tau_ij = np.array([taus[n][i, j] for n in sites])
                    T_l[l, i, j] = np.mean(tau_ij * phases)

        results[r] = T_l

    return results


def fourier_analyze_frame_transport(taus, N, n_sublattices=4):
    """
    Fourier-analyze the frame transport matrices U_{n,n+1} and U_{n,n-1}.
    """
    results_fwd = {}
    results_bwd = {}
    for r in range(n_sublattices):
        sites = np.arange(r, N, n_sublattices)
        N_r = len(sites)
        phi_r = sites * THETA_BC

        n_harmonics = N_r // 2
        U_fwd_l = np.zeros((n_harmonics, 4, 4), dtype=complex)
        U_bwd_l = np.zeros((n_harmonics, 4, 4), dtype=complex)

        for l in range(n_harmonics):
            phases = np.exp(-1j * l * phi_r)
            for i in range(4):
                for j in range(4):
                    u_fwd_ij = np.array([
                        frame_transport(taus[n], taus[(n+1) % N])[i, j]
                        for n in sites
                    ])
                    u_bwd_ij = np.array([
                        frame_transport(taus[n], taus[(n-1) % N])[i, j]
                        for n in sites
                    ])
                    U_fwd_l[l, i, j] = np.mean(u_fwd_ij * phases)
                    U_bwd_l[l, i, j] = np.mean(u_bwd_ij * phases)

        results_fwd[r] = U_fwd_l
        results_bwd[r] = U_bwd_l

    return results_fwd, results_bwd


# ---- Walk operator with periodic BCs ----

def build_walk_periodic(N, taus, mix_phi):
    """Build full walk operator with periodic BCs."""
    dim = 4 * N
    S = np.zeros((dim, dim), dtype=complex)
    for n in range(N):
        n_fwd = (n + 1) % N
        n_bwd = (n - 1) % N
        Pp = proj_plus(taus[n])
        Pm = proj_minus(taus[n])
        U_fwd = frame_transport(taus[n], taus[n_fwd])
        U_bwd = frame_transport(taus[n], taus[n_bwd])
        S[4*n_fwd:4*n_fwd+4, 4*n:4*n+4] += U_fwd @ Pp
        S[4*n_bwd:4*n_bwd+4, 4*n:4*n+4] += U_bwd @ Pm

    V = np.eye(dim, dtype=complex)
    if mix_phi != 0:
        cp, sp = np.cos(mix_phi), np.sin(mix_phi)
        for n in range(N):
            Pp = proj_plus(taus[n])
            Pm = proj_minus(taus[n])
            pp_basis = np.zeros((4, 2), dtype=complex)
            pm_basis = np.zeros((4, 2), dtype=complex)
            np_found = nm_found = 0
            for col in range(4):
                if np_found >= 2 and nm_found >= 2:
                    break
                if np_found < 2:
                    v = Pp[:, col].copy()
                    for j in range(np_found):
                        v -= np.vdot(pp_basis[:, j], v) * pp_basis[:, j]
                    nm = np.real(np.vdot(v, v))
                    if nm > 1e-10:
                        pp_basis[:, np_found] = v / np.sqrt(nm)
                        np_found += 1
                if nm_found < 2:
                    v = Pm[:, col].copy()
                    for j in range(nm_found):
                        v -= np.vdot(pm_basis[:, j], v) * pm_basis[:, j]
                    nm = np.real(np.vdot(v, v))
                    if nm > 1e-10:
                        pm_basis[:, nm_found] = v / np.sqrt(nm)
                        nm_found += 1
            M = np.zeros((4, 4), dtype=complex)
            for j in range(2):
                M += np.outer(pm_basis[:, j], pp_basis[:, j].conj())
                M += np.outer(pp_basis[:, j], pm_basis[:, j].conj())
            Vmix = cp * np.eye(4) + 1j * sp * M
            V[4*n:4*n+4, 4*n:4*n+4] = Vmix

    return V @ S


def quasi_bloch_quality(N, taus, mix_phi):
    """Compute median R² and max relative deviation for given parameters."""
    W = build_walk_periodic(N, taus, mix_phi)
    eigenvalues, eigenvectors = np.linalg.eig(W)
    E = np.angle(eigenvalues)

    phi = (np.arange(N) * THETA_BC) % (2 * np.pi)
    sublat = np.arange(N) % 4

    ipr = np.zeros(len(E))
    for j in range(len(E)):
        psi = eigenvectors[:, j].reshape(N, 4)
        prob = np.sum(np.abs(psi)**2, axis=1)
        prob /= prob.sum()
        ipr[j] = 1.0 / np.sum(prob**2)

    extended = ipr > N / 10
    ext_idx = np.where(extended)[0]

    R2_vals = []
    max_devs = []
    for j in ext_idx:
        psi = eigenvectors[:, j].reshape(N, 4)
        # Extract k
        freqs = np.fft.fftfreq(N) * 2 * np.pi
        total_power = np.zeros(N)
        for a in range(4):
            ft = np.fft.fft(psi[:, a])
            total_power += np.abs(ft)**2
        total_power[0] = 0
        peak = np.argmax(total_power)
        k = freqs[peak]

        # Bloch function
        ns = np.arange(N)
        u = psi * np.exp(-1j * k * ns)[:, None]

        # R² for each sublattice
        r2_list = []
        for r in range(4):
            mask = sublat == r
            u_r = u[mask]
            phi_r = phi[mask]
            u_norm2 = np.sum(np.abs(u_r)**2, axis=1)
            mean_u2 = np.mean(u_norm2)
            if mean_u2 < 1e-15:
                continue
            delta = u_norm2 / mean_u2 - 1
            total_var = np.var(delta)
            if total_var < 1e-15:
                r2_list.append(1.0)
                continue
            n_harm = 6
            design = np.zeros((len(phi_r), 2 * n_harm))
            for h in range(n_harm):
                design[:, 2*h] = np.cos((h+1) * phi_r)
                design[:, 2*h+1] = np.sin((h+1) * phi_r)
            coeffs, _, _, _ = np.linalg.lstsq(design, delta, rcond=None)
            fitted = design @ coeffs
            r2_list.append(np.var(fitted) / total_var)

        if r2_list:
            R2_vals.append(np.mean(r2_list))

        u_norm2 = np.sum(np.abs(u)**2, axis=1)
        mean_u2 = np.mean(u_norm2)
        if mean_u2 > 1e-15:
            max_devs.append(np.max(np.abs(u_norm2 / mean_u2 - 1)))

    return np.median(R2_vals), np.mean(max_devs), len(ext_idx)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400

    print(f"Chain: N={N}")
    print(f"θ = arccos(-2/3) = {THETA_BC:.6f} rad = {np.degrees(THETA_BC):.3f}°")

    print("\nBuilding chain...")
    taus = build_taus(N)

    # ============================================================
    # Part 1: Fourier spectrum of τ operators
    # ============================================================
    print("\n=== Fourier analysis of τ(φ) ===")
    tau_fourier = fourier_analyze_tau(taus, N)

    # Compute Frobenius norm of each harmonic
    print("\nFrobenius norm of τ Fourier coefficients by sublattice:")
    print(f"{'l':>4}  {'r=0':>10}  {'r=1':>10}  {'r=2':>10}  {'r=3':>10}")
    n_show = 20
    all_norms = {}
    for r in range(4):
        T = tau_fourier[r]
        norms = np.array([np.linalg.norm(T[l]) for l in range(min(n_show, len(T)))])
        all_norms[r] = norms
    for l in range(n_show):
        vals = [all_norms[r][l] if l < len(all_norms[r]) else 0 for r in range(4)]
        print(f"{l:4d}  {vals[0]:10.6f}  {vals[1]:10.6f}  {vals[2]:10.6f}  {vals[3]:10.6f}")

    # ============================================================
    # Part 2: Fourier spectrum of frame transport
    # ============================================================
    print("\n=== Fourier analysis of frame transport U_{n,n+1} ===")
    ft_fwd, ft_bwd = fourier_analyze_frame_transport(taus, N)

    print("\nFrobenius norm of U_fwd Fourier coefficients:")
    print(f"{'l':>4}  {'r=0':>10}  {'r=1':>10}  {'r=2':>10}  {'r=3':>10}")
    fwd_norms = {}
    for r in range(4):
        U = ft_fwd[r]
        norms = np.array([np.linalg.norm(U[l]) for l in range(min(n_show, len(U)))])
        fwd_norms[r] = norms
    for l in range(n_show):
        vals = [fwd_norms[r][l] if l < len(fwd_norms[r]) else 0 for r in range(4)]
        print(f"{l:4d}  {vals[0]:10.6f}  {vals[1]:10.6f}  {vals[2]:10.6f}  {vals[3]:10.6f}")

    # ============================================================
    # Part 3: φ_mix sweep — quasi-Bloch quality vs mass
    # ============================================================
    N_sweep = 200  # smaller N for speed
    print(f"\n=== φ_mix sweep (N={N_sweep}) ===")
    taus_sweep = build_taus(N_sweep)

    phi_values = [0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5]
    sweep_results = []
    for phi_mix in phi_values:
        print(f"  φ_mix = {phi_mix:.2f}...", end='', flush=True)
        r2, maxdev, n_ext = quasi_bloch_quality(N_sweep, taus_sweep, phi_mix)
        print(f" R²={r2:.4f}, max_dev={maxdev:.4f}, n_ext={n_ext}")
        sweep_results.append((phi_mix, r2, maxdev, n_ext))

    # ============================================================
    # Plots
    # ============================================================
    fig = plt.figure(figsize=(18, 14))

    # Panel 1: τ Fourier spectrum (log scale)
    ax1 = fig.add_subplot(2, 3, 1)
    for r in range(4):
        norms = all_norms[r]
        ax1.semilogy(range(len(norms)), norms, 'o-', ms=3, label=f'r={r}')
    ax1.set_xlabel('Harmonic index l')
    ax1.set_ylabel('||T_l|| (Frobenius)')
    ax1.set_title('Fourier spectrum of τ(φ)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, n_show - 0.5)

    # Panel 2: Frame transport Fourier spectrum
    ax2 = fig.add_subplot(2, 3, 2)
    for r in range(4):
        norms = fwd_norms[r]
        ax2.semilogy(range(len(norms)), norms, 'o-', ms=3, label=f'r={r}')
    ax2.set_xlabel('Harmonic index l')
    ax2.set_ylabel('||U_l|| (Frobenius)')
    ax2.set_title('Fourier spectrum of U_{fwd}(φ)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, n_show - 0.5)

    # Panel 3: Cumulative spectral weight — fraction of ||τ||² in first M harmonics
    ax3 = fig.add_subplot(2, 3, 3)
    for r in range(4):
        norms = all_norms[r]
        total = np.sum(norms**2)
        cumulative = np.cumsum(norms**2) / total
        ax3.plot(range(len(cumulative)), cumulative, 'o-', ms=3, label=f'r={r}')
    ax3.axhline(0.99, color='k', linestyle='--', alpha=0.5, label='99%')
    ax3.axhline(0.999, color='k', linestyle=':', alpha=0.5, label='99.9%')
    ax3.set_xlabel('Number of harmonics M')
    ax3.set_ylabel('Cumulative ||T||² fraction')
    ax3.set_title('How many harmonics needed?')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: τ matrix elements vs φ — show quasiperiodic structure directly
    ax4 = fig.add_subplot(2, 3, 4)
    phi_sites = (np.arange(N) * THETA_BC) % (2 * np.pi)
    sublat = np.arange(N) % 4
    # Plot Re(τ[0,3]) — an off-diagonal element showing the alpha_1 component
    for r in range(4):
        mask = sublat == r
        phi_r = phi_sites[mask]
        tau_03 = np.array([np.real(taus[n][0, 3]) for n in np.where(mask)[0]])
        order = np.argsort(phi_r)
        ax4.plot(phi_r[order], tau_03[order], '.', ms=2, label=f'r={r}')
    ax4.set_xlabel('φ = nθ mod 2π')
    ax4.set_ylabel('Re(τ[0,3])')
    ax4.set_title('τ matrix element vs perp-space coord')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # Panel 5: φ_mix sweep — R² vs φ_mix
    ax5 = fig.add_subplot(2, 3, 5)
    phi_vals = [s[0] for s in sweep_results]
    r2_vals = [s[1] for s in sweep_results]
    maxdev_vals = [s[2] for s in sweep_results]
    ax5.plot(phi_vals, r2_vals, 'bo-', label='median R²')
    ax5.set_xlabel('φ_mix (mass parameter)')
    ax5.set_ylabel('Median R²')
    ax5.set_title('Quasi-Bloch quality vs mass')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: φ_mix sweep — max deviation vs φ_mix
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(phi_vals, maxdev_vals, 'ro-', label='mean max |u|² dev')
    ax6.set_xlabel('φ_mix (mass parameter)')
    ax6.set_ylabel('Mean max relative deviation')
    ax6.set_title('Bloch function uniformity vs mass')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f'Perpendicular-space Fourier analysis: N={N}\n'
                 f'θ = arccos(-2/3), tet/turn = {2*np.pi/THETA_BC:.4f}', fontsize=14)
    plt.tight_layout()

    out = '/tmp/quasi_bloch_fourier.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    # Summary
    print("\n=== Summary ===")
    for r in range(4):
        norms = all_norms[r]
        total = np.sum(norms**2)
        for M in [1, 2, 3, 5, 10]:
            frac = np.sum(norms[:M]**2) / total
            if M <= len(norms):
                print(f"  Sublattice r={r}: M={M} harmonics capture {frac*100:.2f}% of ||τ||²")


if __name__ == '__main__':
    main()
