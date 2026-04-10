#!/usr/bin/env python3
"""
Quasi-Bloch decomposition of walk eigenstates via cut-and-project.

The BC helix is a 1D quasicrystal: the τ operators vary quasiperiodically
with site index n, governed by the irrational twist angle θ = arccos(-2/3)
per tetrahedron, plus a period-4 sublattice from the face pattern [1,3,0,2].

The cut-and-project Bloch ansatz is:

    ψ(n) = e^{ikn} · F_{n mod 4}(φ(n))

where k is the quasi-momentum, φ(n) = nθ mod 2π is the perpendicular-space
coordinate (accumulated twist angle), and F_r are four smooth functions on
the circle — the Bloch envelopes.

This script tests this ansatz by:
1. Building the walk operator with periodic BCs
2. Diagonalizing and selecting extended eigenstates
3. For each eigenstate, extracting k and computing u(n) = e^{-ikn} ψ(n)
4. Plotting u vs φ, colored by sublattice r = n mod 4
5. Measuring correlation of |u|² deviations with φ
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.helix_geometry import build_taus, centroid, exit_direction

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


# ---- Walk operator with PERIODIC BCs ----

def build_walk_operator(N, taus, mix_phi):
    """Build 4N x 4N walk operator W = V · S with periodic BCs."""
    dim = 4 * N

    # Shift operator (periodic BCs)
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

    # V-mixing operator
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


# ---- Perpendicular space coordinate ----

THETA_BC = np.arccos(-2/3)  # twist per tetrahedron ≈ 2.3005 rad

def perp_coordinate(N):
    """
    Compute perpendicular-space coordinate φ(n) = nθ mod 2π.

    This is the accumulated twist angle around the helix axis.
    """
    ns = np.arange(N)
    return (ns * THETA_BC) % (2 * np.pi)


# ---- Analysis ----

def extract_k_and_bloch(psi_2d, N):
    """
    Extract dominant momentum k and Bloch function u(n) = e^{-ikn} ψ(n).
    """
    freqs = np.fft.fftfreq(N) * 2 * np.pi
    total_power = np.zeros(N)
    for a in range(4):
        ft = np.fft.fft(psi_2d[:, a])
        total_power += np.abs(ft)**2
    total_power[0] = 0

    peak = np.argmax(total_power)
    k = freqs[peak]

    ns = np.arange(N)
    phase = np.exp(-1j * k * ns)
    u = psi_2d * phase[:, None]

    return k, u


def fourier_smoothness(u_sublat, phi_sublat):
    """
    Measure smoothness of u(φ) by computing the ratio of low-frequency
    to total Fourier power when u is resampled onto a regular φ grid.

    Since φ values are irregularly spaced (quasiperiodic), we use a
    different approach: compute the correlation between |u|² and
    low-order Fourier modes of φ.

    Returns the R² of a fit to the first few Fourier harmonics of φ.
    """
    if len(phi_sublat) < 10:
        return 0.0

    # Compute |u|² deviations from mean
    u_norm2 = np.sum(np.abs(u_sublat)**2, axis=1)
    mean_u2 = np.mean(u_norm2)
    if mean_u2 < 1e-15:
        return 0.0
    delta_u2 = u_norm2 / mean_u2 - 1  # relative deviation

    total_var = np.var(delta_u2)
    if total_var < 1e-15:
        return 1.0  # perfectly uniform = perfectly smooth

    # Fit to first n_harmonics Fourier modes of φ
    n_harmonics = 6
    design = np.zeros((len(phi_sublat), 2 * n_harmonics))
    for h in range(n_harmonics):
        design[:, 2*h] = np.cos((h+1) * phi_sublat)
        design[:, 2*h+1] = np.sin((h+1) * phi_sublat)

    # Least squares fit
    coeffs, residuals, _, _ = np.linalg.lstsq(design, delta_u2, rcond=None)
    fitted = design @ coeffs
    explained_var = np.var(fitted)

    return explained_var / total_var  # R²


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    phi_mix = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08

    print(f"Chain: N={N}, phi_mix={phi_mix}")
    print(f"Twist per step θ = arccos(-2/3) = {np.degrees(THETA_BC):.4f}°")
    print(f"Tetrahedra per turn = 2π/θ = {2*np.pi/THETA_BC:.6f} (1+√3 = {1+np.sqrt(3):.6f})")

    print("Building chain...")
    taus = build_taus(N)
    positions = centroid(np.arange(N))
    exit_dirs = np.array([exit_direction(n) for n in range(N)])

    # Perpendicular space coordinate
    phi = perp_coordinate(N)
    sublat = np.arange(N) % 4

    # Verify: compare computed phi with actual azimuthal angles from geometry
    # (to make sure nθ mod 2π is the right coordinate)
    from numpy.linalg import svd
    centered = positions - positions.mean(axis=0)
    _, _, Vt = svd(centered, full_matrices=False)
    helix_axis = Vt[0]
    if helix_axis[2] < 0:
        helix_axis = -helix_axis

    # Azimuthal angles from actual geometry
    ref = np.array([1, 0, 0]) if abs(helix_axis[0]) < 0.9 else np.array([0, 1, 0])
    e_x = ref - np.dot(ref, helix_axis) * helix_axis
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(helix_axis, e_x)

    phi_geom = np.zeros(N)
    for n in range(N):
        d = exit_dirs[n]
        d_perp = d - np.dot(d, helix_axis) * helix_axis
        if np.linalg.norm(d_perp) > 1e-10:
            phi_geom[n] = np.arctan2(np.dot(d_perp, e_y), np.dot(d_perp, e_x))

    # Check correlation between phi and phi_geom (should be linear mod 2π)
    phi_geom_unwrap = np.unwrap(phi_geom)
    phi_ideal_unwrap = np.arange(N) * THETA_BC  # not mod 2π
    # Linear regression
    slope = np.polyfit(phi_ideal_unwrap, phi_geom_unwrap, 1)[0]
    print(f"Correlation slope (ideal vs geom twist): {slope:.6f} (should be ±1)")

    # ---- Walk operator ----
    print(f"\nBuilding walk operator ({4*N}x{4*N}) with periodic BCs...")
    W = build_walk_operator(N, taus, phi_mix)

    # Verify unitarity
    WdW = W.conj().T @ W
    unitarity_err = np.linalg.norm(WdW - np.eye(4*N))
    print(f"Unitarity error: {unitarity_err:.2e}")

    print("Diagonalizing...")
    eigenvalues, eigenvectors = np.linalg.eig(W)
    E = np.angle(eigenvalues)

    # IPR filter
    ipr = np.zeros(len(E))
    for j in range(len(E)):
        psi = eigenvectors[:, j].reshape(N, 4)
        prob = np.sum(np.abs(psi)**2, axis=1)
        prob /= prob.sum()
        ipr[j] = 1.0 / np.sum(prob**2)

    extended = ipr > N / 10
    print(f"Extended states: {extended.sum()} / {len(E)}")

    # ---- Quasi-Bloch analysis ----
    print("\nExtracting quasi-Bloch decomposition...")

    ext_indices = np.where(extended)[0]
    n_ext = len(ext_indices)

    k_vals = np.zeros(n_ext)
    E_vals = np.zeros(n_ext)
    R2_vals = np.zeros(n_ext)  # R² for each sublattice, averaged
    max_rel_dev = np.zeros(n_ext)  # max relative deviation of |u|²

    for i, j in enumerate(ext_indices):
        psi = eigenvectors[:, j].reshape(N, 4)
        k, u = extract_k_and_bloch(psi, N)
        k_vals[i] = k
        E_vals[i] = E[j]

        # Compute R² for each sublattice
        r2_list = []
        for r in range(4):
            mask = sublat == r
            r2 = fourier_smoothness(u[mask], phi[mask])
            r2_list.append(r2)
        R2_vals[i] = np.mean(r2_list)

        # Max relative deviation of |u|²
        u_norm2 = np.sum(np.abs(u)**2, axis=1)
        mean_u2 = np.mean(u_norm2)
        if mean_u2 > 1e-15:
            max_rel_dev[i] = np.max(np.abs(u_norm2 / mean_u2 - 1))

    print(f"Median R² (|u|² explained by Fourier harmonics of φ): {np.median(R2_vals):.4f}")
    print(f"Mean max relative deviation of |u|²: {np.mean(max_rel_dev):.4f}")

    # ---- Select example states ----
    # Pick states at different energies from positive branch
    pos_mask = E_vals > 0.01
    pos_idx = np.where(pos_mask)[0]
    if len(pos_idx) > 0:
        sorted_by_E = pos_idx[np.argsort(np.abs(E_vals[pos_idx]))]
        n_examples = min(6, len(sorted_by_E))
        picks = sorted_by_E[np.linspace(0, len(sorted_by_E)-1, n_examples, dtype=int)]
    else:
        picks = np.arange(min(6, n_ext))

    # ---- Plots ----
    fig = plt.figure(figsize=(18, 16))
    colors_sub = ['C0', 'C1', 'C2', 'C3']

    # Panel 1: φ(n) mod 2π — perpendicular space coordinate
    ax1 = fig.add_subplot(4, 3, 1)
    for r in range(4):
        mask = sublat == r
        ax1.scatter(np.where(mask)[0][:100], phi[mask][:mask.sum().clip(max=100)],
                    s=4, c=colors_sub[r], label=f'r={r}')
    ax1.set_xlabel('Site n')
    ax1.set_ylabel('φ = nθ mod 2π')
    ax1.set_title('Perp-space coordinate')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: φ(n) fills [0, 2π) densely
    ax2 = fig.add_subplot(4, 3, 2)
    # For one sublattice, show sorted φ values — should be nearly uniform
    r = 0
    mask = sublat == r
    phi_sorted = np.sort(phi[mask])
    expected = np.linspace(0, 2*np.pi, mask.sum(), endpoint=False)
    ax2.plot(expected, phi_sorted, '.', ms=2)
    ax2.plot([0, 2*np.pi], [0, 2*np.pi], 'r--', alpha=0.5)
    ax2.set_xlabel('Expected (uniform)')
    ax2.set_ylabel('Sorted φ values')
    ax2.set_title(f'Equidistribution test (r={r})')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Dispersion with R² coloring
    ax3 = fig.add_subplot(4, 3, 3)
    sc = ax3.scatter(np.abs(k_vals), np.abs(E_vals), s=4, c=R2_vals,
                     cmap='RdYlGn', vmin=0, vmax=0.5, alpha=0.6)
    k_th = np.linspace(0, np.pi, 200)
    m_dirac = 0.9 * phi_mix
    ax3.plot(k_th, np.sqrt(k_th**2 + m_dirac**2), 'r-', lw=2, label=f'Dirac m={m_dirac:.3f}')
    ax3.set_xlabel('|k|')
    ax3.set_ylabel('|E|')
    ax3.set_title('Dispersion (color = R²)')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, np.pi)
    ax3.set_ylim(0, np.pi)
    plt.colorbar(sc, ax=ax3, label='R²')
    ax3.grid(True, alpha=0.3)

    # Panels 4-9: Bloch functions — relative deviation of |u|² vs φ
    for pi_idx, pick in enumerate(picks[:6]):
        ax = fig.add_subplot(4, 3, 4 + pi_idx)
        j = ext_indices[pick]
        psi = eigenvectors[:, j].reshape(N, 4)
        k, u = extract_k_and_bloch(psi, N)
        u_norm2 = np.sum(np.abs(u)**2, axis=1)
        mean_u2 = np.mean(u_norm2)
        rel_dev = u_norm2 / mean_u2 - 1  # relative deviation from mean

        for r in range(4):
            mask = sublat == r
            phi_r = phi[mask]
            order = np.argsort(phi_r)
            ax.plot(phi_r[order], rel_dev[mask][order], '.', ms=3,
                    c=colors_sub[r], label=f'r={r}' if pi_idx == 0 else None)

        ax.axhline(0, color='k', alpha=0.3)
        ax.set_xlabel('φ (perp-space)')
        ax.set_ylabel('|u|²/<|u|²> − 1')
        Ej = E_vals[pick]
        kj = k_vals[pick]
        r2j = R2_vals[pick]
        ax.set_title(f'E={Ej:.3f}, k={kj:.3f}, R²={r2j:.3f}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if pi_idx == 0:
            ax.legend(fontsize=6)

    # Panel 10: R² distribution
    ax10 = fig.add_subplot(4, 3, 10)
    ax10.hist(R2_vals, bins=30, alpha=0.7, edgecolor='black')
    ax10.axvline(np.median(R2_vals), color='r', linestyle='--',
                 label=f'median={np.median(R2_vals):.3f}')
    ax10.set_xlabel('R² (variance explained by φ-harmonics)')
    ax10.set_ylabel('Count')
    ax10.set_title('Quasi-Bloch quality')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Panel 11: R² vs |E|
    ax11 = fig.add_subplot(4, 3, 11)
    ax11.scatter(np.abs(E_vals), R2_vals, s=4, alpha=0.5)
    ax11.set_xlabel('|E|')
    ax11.set_ylabel('R²')
    ax11.set_title('Bloch quality vs energy')
    ax11.grid(True, alpha=0.3)

    # Panel 12: Max relative deviation vs |E|
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.scatter(np.abs(E_vals), max_rel_dev, s=4, alpha=0.5)
    ax12.set_xlabel('|E|')
    ax12.set_ylabel('max |u|² / <|u|²> − 1')
    ax12.set_title('Max density deviation')
    ax12.grid(True, alpha=0.3)

    fig.suptitle(f'Quasi-Bloch analysis: N={N}, φ_mix={phi_mix}, periodic BCs\n'
                 f'ψ(n) = e^{{ikn}} F_{{n mod 4}}(nθ mod 2π),  '
                 f'median R² = {np.median(R2_vals):.4f}', fontsize=13)
    plt.tight_layout()

    out = '/tmp/quasi_bloch.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    # ---- Spinor component detail for best-R² state ----
    best_pick = ext_indices[np.argmax(R2_vals)]
    psi_best = eigenvectors[:, best_pick].reshape(N, 4)
    k_best, u_best = extract_k_and_bloch(psi_best, N)
    E_best = E[best_pick]
    R2_best = np.max(R2_vals)

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    for comp in range(4):
        ax = axes2[comp // 2, comp % 2]
        for r in range(4):
            mask = sublat == r
            phi_r = phi[mask]
            order = np.argsort(phi_r)
            u_comp = u_best[mask, comp]
            ax.plot(phi_r[order], np.real(u_comp[order]), '.', ms=3,
                    c=colors_sub[r], label=f'Re r={r}' if comp == 0 else None)
            ax.plot(phi_r[order], np.imag(u_comp[order]), 'x', ms=2,
                    c=colors_sub[r], alpha=0.4)
        ax.set_xlabel('φ = nθ mod 2π')
        ax.set_ylabel(f'u_{comp}')
        ax.set_title(f'Spinor component {comp}')
        ax.grid(True, alpha=0.3)
    axes2[0, 0].legend(fontsize=6)
    fig2.suptitle(f'Best-R² Bloch function: E={E_best:.3f}, k={k_best:.3f}, R²={R2_best:.3f}',
                  fontsize=13)
    plt.tight_layout()

    out2 = '/tmp/quasi_bloch_best.png'
    plt.savefig(out2, dpi=150)
    print(f"Best-R² component plot saved to {out2}")


if __name__ == '__main__':
    main()
