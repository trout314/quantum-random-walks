#!/usr/bin/env python3
"""
Truncated quasi-Bloch operator: band structure from a tiny matrix.

The walk operator has two kinds of position-dependence:
1. Period-4 sublattice structure from face pattern [1,3,0,2]
2. Quasiperiodic modulation with irrational twist θ = arccos(-2/3)

The proper quasi-Bloch basis is:
  ψ(n) = e^{ikn} Σ_{m,r} c_{m,r,s} e^{imnθ} e^{i·2πrn/4}

where m ∈ {-M,...,M} is the quasiperiodic harmonic, r ∈ {0,1,2,3} the
sublattice Fourier index, and s ∈ {0,...,3} the spinor index.

Equivalently, the site-dependent walk blocks have Fourier expansion:
  F_n = Σ_{l,r} F̂_{l,r} e^{ilnθ} e^{i·2πrn/4}

Truncating to M quasiperiodic harmonics gives a matrix of size
4(sublattice) × (2M+1)(harmonics) × 4(spinor) = 16(2M+1), independent of N.

For M=1: 48×48.  For M=2: 80×80.  Compare to full: 4N×4N = 1600×1600.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


# ---- Geometry & algebra ----

def init_tet():
    return np.array([
        [0, 0, 1],
        [2*np.sqrt(2)/3, 0, -1/3],
        [-np.sqrt(2)/3, np.sqrt(6)/3, -1/3],
        [-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3],
    ])

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def helix_step(pos, dirs, face):
    e = dirs[face].copy()
    pos += e * (-2/3)
    for a in range(4):
        dirs[a] = reflect(dirs[a], e)

def reorth(dirs):
    m = dirs.mean(axis=0)
    dirs -= m
    for a in range(4):
        nm = np.linalg.norm(dirs[a])
        if nm > 1e-15:
            dirs[a] /= nm

def alpha_mat(idx):
    m = np.zeros((4, 4), dtype=complex)
    if idx == 0:
        m[0,3] = m[1,2] = m[2,1] = m[3,0] = 1
    elif idx == 1:
        m[0,3] = -1j; m[1,2] = 1j; m[2,1] = -1j; m[3,0] = 1j
    elif idx == 2:
        m[0,2] = 1; m[1,3] = -1; m[2,0] = 1; m[3,1] = -1
    return m

def make_tau(d):
    nu = np.sqrt(7) / 4
    tau = np.diag([nu, nu, -nu, -nu]).astype(complex)
    for a in range(3):
        tau += 0.75 * d[a] * alpha_mat(a)
    return tau

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


def build_chain_taus(N, pat):
    dirs_all = np.zeros((N, 4, 3))
    taus = np.zeros((N, 4, 4), dtype=complex)
    dirs = init_tet()
    pos = np.zeros(3)
    for n in range(N):
        if n == 0:
            dirs_all[0] = dirs.copy()
        else:
            dirs = dirs_all[n-1].copy()
            helix_step(pos, dirs, pat[(n-1) % 4])
            if n % 8 == 0:
                reorth(dirs)
            dirs_all[n] = dirs
        face = pat[n % 4]
        taus[n] = make_tau(dirs_all[n][face])
    return taus


def build_vmix_block(tau, mix_phi):
    if mix_phi == 0:
        return np.eye(4, dtype=complex)
    cp, sp = np.cos(mix_phi), np.sin(mix_phi)
    Pp = proj_plus(tau)
    Pm = proj_minus(tau)
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
    return cp * np.eye(4) + 1j * sp * M


def compute_walk_blocks(N, taus, mix_phi):
    """
    Walk operator: (Wψ)(m) = F_m ψ(m-1) + G_m ψ(m+1)
    F_m = V_m · U_{(m-1)→m} · P⁺_{m-1}
    G_m = V_m · U_{(m+1)→m} · P⁻_{m+1}
    """
    F = np.zeros((N, 4, 4), dtype=complex)
    G = np.zeros((N, 4, 4), dtype=complex)
    for m in range(N):
        V_m = build_vmix_block(taus[m], mix_phi)
        m_prev = (m - 1) % N
        m_next = (m + 1) % N
        U_fwd = frame_transport(taus[m_prev], taus[m])
        U_bwd = frame_transport(taus[m_next], taus[m])
        Pp_prev = proj_plus(taus[m_prev])
        Pm_next = proj_minus(taus[m_next])
        F[m] = V_m @ U_fwd @ Pp_prev
        G[m] = V_m @ U_bwd @ Pm_next
    return F, G


def fourier_coefficients_2d(blocks, N, n_qp_harmonics):
    """
    Compute 2D Fourier coefficients: quasiperiodic (nθ) × periodic (2πn/4).

    blocks[n] = Σ_{l,r} B̂_{l,r} e^{ilnθ} e^{i·2πrn/4}

    Returns array of shape (2*n_qp_harmonics+1, 4, 4, 4) where the
    indices are (l_idx, r, spinor_i, spinor_j).
    """
    M = n_qp_harmonics
    n_sub = 4
    B_hat = np.zeros((2 * M + 1, n_sub, 4, 4), dtype=complex)

    ns = np.arange(N)
    for l_idx, l in enumerate(range(-M, M + 1)):
        for r in range(n_sub):
            # Phase: e^{-ilnθ} e^{-i·2πrn/4}
            phase = np.exp(-1j * l * ns * THETA_BC) * np.exp(-1j * 2 * np.pi * r * ns / n_sub)
            for i in range(4):
                for j in range(4):
                    B_hat[l_idx, r, i, j] = np.mean(blocks[:, i, j] * phase)

    return B_hat


def build_truncated_Wk(k, F_hat, G_hat, n_qp_harmonics):
    """
    Build truncated W(k) in the combined (m, r, s) basis.

    State index: (m, r, s) where
      m ∈ {-M,...,M}  quasiperiodic harmonic
      r ∈ {0,1,2,3}   sublattice Fourier index
      s ∈ {0,...,3}    spinor

    dim = 4 × (2M+1) × 4 = 16(2M+1)

    The walk operator:
      (Wψ)(n) = F_n ψ(n-1) + G_n ψ(n+1)

    In the Bloch basis ψ(n) = e^{ikn} u(n):
      (W_k u)(n) = e^{-ik} F_n u(n-1) + e^{ik} G_n u(n+1)

    With u(n) = Σ_{m,r} c_{m,r} e^{imnθ} e^{i2πrn/4}:

    The shift u(n-1) brings phase e^{-imθ} e^{-i2πr/4}
    F_n has Fourier components F̂_{l,q} contributing e^{ilnθ} e^{i2πqn/4}

    Coupling: F̂_{l,q} takes (m, r) → (m+l, (r+q) mod 4)
    with extra shift phases e^{-imθ} e^{-i2πr/4} from the u(n-1).

    Full matrix element:
    [W(k)]_{m',r'; m,r} = e^{-ik} · e^{-imθ} · e^{-i2πr/4} · F̂_{m'-m, (r'-r) mod 4}
                         + e^{+ik} · e^{+imθ} · e^{+i2πr/4} · Ĝ_{m'-m, (r'-r) mod 4}
    """
    M = n_qp_harmonics
    n_sub = 4
    n_qp = 2 * M + 1
    dim = n_sub * n_qp * 4  # (r, m, spinor)

    W = np.zeros((dim, dim), dtype=complex)

    def idx(m, r, s):
        """Map (m, r, s) -> flat index."""
        return ((m + M) * n_sub + r) * 4 + s

    for m_prime in range(-M, M + 1):
        for r_prime in range(n_sub):
            for m in range(-M, M + 1):
                for r in range(n_sub):
                    delta_m = m_prime - m
                    delta_r = (r_prime - r) % n_sub

                    # Check if delta_m is within our Fourier coefficient range
                    l_idx = delta_m + M
                    if not (0 <= l_idx < n_qp):
                        continue

                    # Forward contribution: shift n-1 → n
                    # Phase from u(n-1): e^{-imθ} e^{-i2πr/4}
                    fwd_phase = np.exp(-1j * k) * np.exp(-1j * m * THETA_BC) * np.exp(-1j * 2 * np.pi * r / n_sub)
                    F_block = F_hat[l_idx, delta_r]  # 4×4 spinor matrix

                    # Backward contribution: shift n+1 → n
                    # Phase from u(n+1): e^{+imθ} e^{+i2πr/4}
                    bwd_phase = np.exp(1j * k) * np.exp(1j * m * THETA_BC) * np.exp(1j * 2 * np.pi * r / n_sub)
                    G_block = G_hat[l_idx, delta_r]  # 4×4 spinor matrix

                    # Fill the 4×4 spinor block
                    for sp in range(4):
                        for s in range(4):
                            row = idx(m_prime, r_prime, sp)
                            col = idx(m, r, s)
                            W[row, col] += fwd_phase * F_block[sp, s] + bwd_phase * G_block[sp, s]

    return W


def build_full_walk(N, taus, mix_phi):
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
        for n in range(N):
            V[4*n:4*n+4, 4*n:4*n+4] = build_vmix_block(taus[n], mix_phi)
    return V @ S


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    mix_phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    M_max = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    pat = [1, 3, 0, 2]
    n_k = 200

    print(f"N={N}, φ_mix={mix_phi}, M_max={M_max}, n_k={n_k}")

    print("Building chain...")
    taus = build_chain_taus(N, pat)

    print("Computing walk blocks...")
    F_blocks, G_blocks = compute_walk_blocks(N, taus, mix_phi)

    # 2D Fourier coefficients
    print(f"Computing 2D Fourier coefficients (M={M_max})...")
    F_hat = fourier_coefficients_2d(F_blocks, N, M_max)
    G_hat = fourier_coefficients_2d(G_blocks, N, M_max)

    # Print norms
    print(f"\n||F̂_{{l,r}}|| (quasiperiodic l × sublattice r):")
    print(f"{'l':>4}", end='')
    for r in range(4):
        print(f"  {'r='+str(r):>8}", end='')
    print()
    for l in range(-M_max, M_max + 1):
        l_idx = l + M_max
        print(f"{l:4d}", end='')
        for r in range(4):
            print(f"  {np.linalg.norm(F_hat[l_idx, r]):8.5f}", end='')
        print()

    # ---- Full diagonalization reference ----
    print(f"\nFull diagonalization ({4*N}×{4*N})...")
    W_full = build_full_walk(N, taus, mix_phi)
    evals_full, evecs_full = np.linalg.eig(W_full)
    E_full = np.angle(evals_full)

    # Extract k for each eigenvalue
    k_full = np.zeros(len(E_full))
    for j in range(len(E_full)):
        psi = evecs_full[:, j].reshape(N, 4)
        freqs = np.fft.fftfreq(N) * 2 * np.pi
        total_power = np.zeros(N)
        for a in range(4):
            ft = np.fft.fft(psi[:, a])
            total_power += np.abs(ft)**2
        total_power[0] = 0
        k_full[j] = freqs[np.argmax(total_power)]

    # ---- Truncated W(k) for different M values ----
    M_values = sorted(set([1, 2, 3, min(5, M_max), M_max]))

    k_points = np.linspace(-np.pi, np.pi, n_k, endpoint=False)

    results = {}
    for M in M_values:
        dim_M = 4 * 4 * (2 * M + 1)
        print(f"\nTruncated W(k): M={M}, dim={dim_M}×{dim_M}")

        F_hat_M = F_hat[M_max - M:M_max + M + 1]
        G_hat_M = G_hat[M_max - M:M_max + M + 1]

        E_bands = np.zeros((n_k, dim_M))
        for i, k in enumerate(k_points):
            Wk = build_truncated_Wk(k, F_hat_M, G_hat_M, M)
            evals_k = np.linalg.eigvals(Wk)
            E_bands[i] = np.sort(np.angle(evals_k))

        results[M] = (k_points, E_bands)

        # Eigenvalue magnitude check (should be near 1 for unitary)
        for i_check in [0, n_k//2]:
            Wk = build_truncated_Wk(k_points[i_check], F_hat_M, G_hat_M, M)
            ev = np.linalg.eigvals(Wk)
            print(f"  k={k_points[i_check]:.2f}: |eigenvalues| range "
                  f"[{np.min(np.abs(ev)):.4f}, {np.max(np.abs(ev)):.4f}]")

    # ============================================================
    # Plots
    # ============================================================
    n_M = len(M_values)
    fig = plt.figure(figsize=(18, 5 * ((n_M + 4) // 3 + 1)))
    n_cols = 3
    n_rows = (n_M + 1 + n_cols - 1) // n_cols + 1

    # Panel 1: Full diag reference
    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.scatter(k_full, E_full, s=1, alpha=0.3, c='blue')
    m_dirac = 0.9 * mix_phi
    k_th = np.linspace(-np.pi, np.pi, 400)
    ax.plot(k_th, np.sqrt(k_th**2 + m_dirac**2), 'r-', lw=1, alpha=0.7)
    ax.plot(k_th, -np.sqrt(k_th**2 + m_dirac**2), 'r-', lw=1, alpha=0.7)
    ax.set_xlabel('k')
    ax.set_ylabel('E')
    ax.set_title(f'Full ({4*N}×{4*N})')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True, alpha=0.3)

    # Truncated panels
    for idx, M in enumerate(M_values):
        ax = fig.add_subplot(n_rows, n_cols, 2 + idx)
        k_pts, E_bands = results[M]
        dim_M = E_bands.shape[1]
        for band in range(dim_M):
            ax.plot(k_pts, E_bands[:, band], '.', ms=0.5, alpha=0.4, c='blue')
        ax.plot(k_th, np.sqrt(k_th**2 + m_dirac**2), 'r-', lw=1, alpha=0.7)
        ax.plot(k_th, -np.sqrt(k_th**2 + m_dirac**2), 'r-', lw=1, alpha=0.7)
        ax.set_xlabel('k')
        ax.set_ylabel('E')
        ax.set_title(f'M={M} ({dim_M}×{dim_M})')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)

    # DOS comparison
    ax_dos = fig.add_subplot(n_rows, n_cols, n_cols * (n_rows - 1) + 1)
    bins = np.linspace(-np.pi, np.pi, 150)
    ax_dos.hist(E_full, bins=bins, density=True, alpha=0.4, label='Full', color='black')
    colors_M = ['C0', 'C1', 'C2', 'C3', 'C4']
    for idx, M in enumerate(M_values):
        _, E_bands = results[M]
        ax_dos.hist(E_bands.flatten(), bins=bins, density=True, alpha=0.3,
                    label=f'M={M}', color=colors_M[idx % len(colors_M)],
                    histtype='step', linewidth=1.5)
    ax_dos.set_xlabel('E')
    ax_dos.set_ylabel('DOS')
    ax_dos.set_title('DOS comparison')
    ax_dos.legend(fontsize=7)
    ax_dos.grid(True, alpha=0.3)

    # Near-gap zoom
    ax_zoom = fig.add_subplot(n_rows, n_cols, n_cols * (n_rows - 1) + 2)
    zoom_E = 5 * max(m_dirac, 0.05)
    ax_zoom.scatter(k_full, E_full, s=3, alpha=0.3, c='black', label='Full', zorder=1)
    M_last = M_values[-1]
    k_pts, E_bands = results[M_last]
    for band in range(E_bands.shape[1]):
        mask = np.abs(E_bands[:, band]) < zoom_E
        if mask.any():
            ax_zoom.plot(k_pts[mask], E_bands[mask, band], '.', ms=2,
                        alpha=0.6, c='C0', label=f'M={M_last}' if band == 0 else None,
                        zorder=2)
    ax_zoom.plot(k_th, np.sqrt(k_th**2 + m_dirac**2), 'r-', lw=1.5)
    ax_zoom.plot(k_th, -np.sqrt(k_th**2 + m_dirac**2), 'r-', lw=1.5)
    ax_zoom.set_xlabel('k')
    ax_zoom.set_ylabel('E')
    ax_zoom.set_title(f'Near-gap zoom (M={M_last})')
    ax_zoom.set_xlim(-3*zoom_E, 3*zoom_E)
    ax_zoom.set_ylim(-zoom_E, zoom_E)
    ax_zoom.legend(fontsize=7)
    ax_zoom.grid(True, alpha=0.3)

    # Fourier spectrum heatmap
    ax_heat = fig.add_subplot(n_rows, n_cols, n_cols * (n_rows - 1) + 3)
    norms_2d = np.zeros((2 * M_max + 1, 4))
    for l_idx in range(2 * M_max + 1):
        for r in range(4):
            norms_2d[l_idx, r] = np.linalg.norm(F_hat[l_idx, r])
    im = ax_heat.imshow(np.log10(norms_2d + 1e-10).T, aspect='auto',
                        extent=[-M_max-0.5, M_max+0.5, -0.5, 3.5],
                        origin='lower', cmap='viridis')
    ax_heat.set_xlabel('Quasiperiodic harmonic l')
    ax_heat.set_ylabel('Sublattice index r')
    ax_heat.set_title('log₁₀ ||F̂_{l,r}||')
    plt.colorbar(im, ax=ax_heat)

    fig.suptitle(f'Truncated quasi-Bloch: N={N}, φ_mix={mix_phi}\n'
                 f'Basis: e^{{ikn}} e^{{imnθ}} e^{{i2πrn/4}}, dim = 16(2M+1)',
                 fontsize=14)
    plt.tight_layout()

    out = '/tmp/quasi_bloch_truncated.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    main()
