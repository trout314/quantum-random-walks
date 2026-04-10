"""
Quasi-Bloch eigenvalue equation on the circle.

The walk operator on the infinite BC helix has the quasi-Bloch ansatz:

    φ(n) = e^{ikn} F(nθ mod 2π)

where F: S¹ → C⁴. The eigenvalue equation Sφ = λφ reduces to:

    λ F(ξ) = e^{-ik} A(ξ) F(ξ-θ) + e^{ik} B(ξ) F(ξ+θ)

where:
    A(ξ) = frame_transport(τ(ξ-θ), τ(ξ)) @ P⁺(ξ-θ)
    B(ξ) = frame_transport(τ(ξ+θ), τ(ξ)) @ P⁻(ξ+θ)

FOURIER-SPACE APPROACH:
Expand F(ξ) = Σ_ℓ c_ℓ e^{iℓξ} with c_ℓ ∈ C⁴.
Then φ(n) = Σ_ℓ c_ℓ e^{i(k+ℓθ)n}, so each Fourier mode ℓ
corresponds to effective momentum p = k + ℓθ.

The eigenvalue equation becomes a matrix equation for {c_ℓ}:
    λ c_m = Σ_ℓ [e^{-ik} â_{m-ℓ} e^{-iℓθ} + e^{ik} b̂_{m-ℓ} e^{iℓθ}] c_ℓ

where â_p, b̂_p are 4×4 Fourier coefficients of A(ξ), B(ξ).
Truncating at |ℓ| ≤ L gives a 4(2L+1) × 4(2L+1) eigenvalue problem.
"""

import numpy as np
import time
from src.helix_geometry import THETA_BC, R_VERTEX, H_VERTEX
from src.walk import make_tau_from_dir, frame_transport, I4


def exit_direction_continuous(xi):
    """
    Compute the exit direction as a continuous function of ξ = nθ.
    Depends only on ξ, not on n mod 4 separately (verified numerically).
    """
    th = THETA_BC
    cos_sum = np.cos(xi + th) + np.cos(xi + 2*th) + np.cos(xi + 3*th)
    sin_sum = np.sin(xi + th) + np.sin(xi + 2*th) + np.sin(xi + 3*th)

    dx = R_VERTEX * (0.75 * np.cos(xi) - 0.25 * cos_sum)
    dy = R_VERTEX * (0.75 * np.sin(xi) - 0.25 * sin_sum)
    dz = -1.5 * H_VERTEX

    d = np.array([dx, dy, dz])
    return d / np.linalg.norm(d)


def compute_AB_fourier(M_fft=1024):
    """
    Compute Fourier coefficients of A(ξ) and B(ξ) via FFT.

    Returns
    -------
    a_hat : ndarray (M_fft, 4, 4) — Fourier coefficients of A
    b_hat : ndarray (M_fft, 4, 4) — Fourier coefficients of B
    """
    th = THETA_BC
    xi_grid = 2 * np.pi * np.arange(M_fft) / M_fft

    A_vals = np.zeros((M_fft, 4, 4), dtype=complex)
    B_vals = np.zeros((M_fft, 4, 4), dtype=complex)

    for m in range(M_fft):
        xi = xi_grid[m]
        d_here = exit_direction_continuous(xi)
        d_prev = exit_direction_continuous(xi - th)
        d_next = exit_direction_continuous(xi + th)

        tau_here = make_tau_from_dir(d_here)
        tau_prev = make_tau_from_dir(d_prev)
        tau_next = make_tau_from_dir(d_next)

        U_fwd = frame_transport(tau_prev, tau_here)
        U_bwd = frame_transport(tau_next, tau_here)
        P_plus_prev = 0.5 * (I4 + tau_prev)
        P_minus_next = 0.5 * (I4 - tau_next)

        A_vals[m] = U_fwd @ P_plus_prev
        B_vals[m] = U_bwd @ P_minus_next

    # FFT each matrix element
    a_hat = np.zeros_like(A_vals)
    b_hat = np.zeros_like(B_vals)
    for i in range(4):
        for j in range(4):
            a_hat[:, i, j] = np.fft.fft(A_vals[:, i, j]) / M_fft
            b_hat[:, i, j] = np.fft.fft(B_vals[:, i, j]) / M_fft

    return a_hat, b_hat


def build_fourier_matrix(k, L, a_hat, b_hat):
    """
    Build the 4(2L+1) × 4(2L+1) eigenvalue matrix in Fourier space.

    For Fourier modes ℓ = -L, ..., L, the matrix element (m, ℓ) is:
        H_{m,ℓ} = e^{-ik} â_{m-ℓ} e^{-iℓθ} + e^{ik} b̂_{m-ℓ} e^{iℓθ}

    where â_p is the p-th Fourier coefficient of A(ξ) (with wrapping).
    """
    th = THETA_BC
    N_modes = 2 * L + 1
    dim = 4 * N_modes
    M_fft = a_hat.shape[0]

    H = np.zeros((dim, dim), dtype=complex)

    eik = np.exp(1j * k)
    emik = np.exp(-1j * k)

    for im, m in enumerate(range(-L, L + 1)):
        for il, ell in enumerate(range(-L, L + 1)):
            p = (m - ell) % M_fft  # Fourier index (wrapped)
            phase_fwd = np.exp(-1j * ell * th)
            phase_bwd = np.exp(1j * ell * th)

            block = emik * a_hat[p] * phase_fwd + eik * b_hat[p] * phase_bwd
            H[4*im:4*im+4, 4*il:4*il+4] = block

    return H


def build_grid_matrix(k, M):
    """
    Build the 4M × 4M eigenvalue matrix on a real-space grid.
    (Kept for comparison with Fourier approach.)
    """
    th = THETA_BC
    s = round(M * th / (2 * np.pi))
    xi_grid = 2 * np.pi * np.arange(M) / M

    H = np.zeros((4*M, 4*M), dtype=complex)
    eik = np.exp(1j * k)
    emik = np.exp(-1j * k)

    for m in range(M):
        xi = xi_grid[m]
        d_here = exit_direction_continuous(xi)
        d_prev = exit_direction_continuous(xi - th)
        d_next = exit_direction_continuous(xi + th)

        tau_here = make_tau_from_dir(d_here)
        tau_prev = make_tau_from_dir(d_prev)
        tau_next = make_tau_from_dir(d_next)

        U_fwd = frame_transport(tau_prev, tau_here)
        U_bwd = frame_transport(tau_next, tau_here)
        P_plus_prev = 0.5 * (I4 + tau_prev)
        P_minus_next = 0.5 * (I4 - tau_next)

        A = U_fwd @ P_plus_prev
        B = U_bwd @ P_minus_next

        m_prev = (m - s) % M
        m_next = (m + s) % M

        H[4*m:4*m+4, 4*m_prev:4*m_prev+4] += emik * A
        H[4*m:4*m+4, 4*m_next:4*m_next+4] += eik * B

    return H


def main():
    print("=" * 70)
    print("Quasi-Bloch Eigenvalue Equation — Fourier Space")
    print("=" * 70)

    # Validate
    print("\n--- Validation ---")
    from src.helix_geometry import exit_direction
    max_err = 0
    for n in range(100):
        xi = n * THETA_BC
        d_cont = exit_direction_continuous(xi)
        d_disc = exit_direction(n)
        err = np.linalg.norm(d_cont - d_disc)
        max_err = max(max_err, err)
    print(f"exit_direction continuous vs discrete: max error = {max_err:.2e}")

    # Compute Fourier coefficients
    print("\n--- Fourier coefficients of A(ξ), B(ξ) ---")
    M_fft = 1024
    t0 = time.time()
    a_hat, b_hat = compute_AB_fourier(M_fft)
    print(f"FFT computed in {time.time()-t0:.3f}s (M_fft={M_fft})")

    # Check decay of Fourier coefficients
    a_mag = np.array([np.linalg.norm(a_hat[p]) for p in range(M_fft)])
    # Reindex: p=0 is DC, p=1,...,M/2 are positive, p=M/2+1,...,M-1 are negative
    # Show |p| = 0, 1, 2, ..., 20
    print(f"\nFourier coefficient magnitudes ||â_p|| (showing |p| = 0..15):")
    print(f"{'|p|':>4s}  {'||â_p||':>12s}  {'||b̂_p||':>12s}")
    b_mag = np.array([np.linalg.norm(b_hat[p]) for p in range(M_fft)])
    for p in range(16):
        if p == 0:
            print(f"{p:4d}  {a_mag[0]:12.6e}  {b_mag[0]:12.6e}")
        else:
            # positive and negative combined
            a_pm = a_mag[p] + a_mag[M_fft - p]
            b_pm = b_mag[p] + b_mag[M_fft - p]
            print(f"{p:4d}  {a_pm:12.6e}  {b_pm:12.6e}")

    # Determine safe truncation
    total_a = sum(a_mag)
    for L_test in [2, 5, 10, 20, 50]:
        captured = sum(a_mag[p] for p in range(L_test+1)) + \
                   sum(a_mag[M_fft-p] for p in range(1, L_test+1))
        print(f"L={L_test:3d}: captured {captured/total_a*100:.4f}% of ||â||")

    # Timing: Fourier approach for various L
    print("\n--- Timing: Fourier eigenvalue problem ---")
    k = 0.5
    for L in [5, 10, 20, 50, 100, 200, 500]:
        dim = 4 * (2*L + 1)
        t0 = time.time()
        H = build_fourier_matrix(k, L, a_hat, b_hat)
        t_build = time.time() - t0

        t0 = time.time()
        evals = np.linalg.eigvals(H)
        t_solve = time.time() - t0

        mags = np.abs(evals)
        mag_err = np.max(np.abs(mags - 1))

        print(f"L={L:4d} (dim={dim:5d}): build={t_build:.4f}s, "
              f"solve={t_solve:.4f}s, max ||λ|-1| = {mag_err:.2e}")

    # Convergence test: eigenvalues at ℓ=0 for increasing L
    print("\n--- Convergence of ℓ=0 eigenvalues ---")
    k = 0.5
    prev_evals = None
    for L in [5, 10, 20, 50, 100]:
        H = build_fourier_matrix(k, L, a_hat, b_hat)
        evals, evecs = np.linalg.eig(H)
        dim = 4 * (2*L + 1)

        # Find the 4 eigenvalues whose eigenvectors are concentrated at ℓ=0
        # ℓ=0 corresponds to index im = L (since modes go from -L to L)
        ell0_weight = np.zeros(dim)
        for i in range(dim):
            v = evecs[:, i]
            # Weight at ℓ=0 block (index L in the mode ordering)
            ell0_block = v[4*L:4*L+4]
            ell0_weight[i] = np.sum(np.abs(ell0_block)**2) / np.sum(np.abs(v)**2)

        # Top 4 by ℓ=0 weight
        top4 = np.argsort(-ell0_weight)[:4]
        phases = np.sort(np.angle(evals[top4]))
        weights = np.sort(ell0_weight[top4])[::-1]

        if prev_evals is not None:
            diff = np.max(np.abs(np.sort(np.angle(evals[top4])) - prev_evals))
        else:
            diff = float('nan')
        prev_evals = phases

        print(f"L={L:4d}: phases/π = [{', '.join(f'{p/np.pi:+.6f}' for p in phases)}]")
        print(f"         ℓ=0 weights = [{', '.join(f'{w:.4f}' for w in weights)}], "
              f"Δ from prev = {diff:.2e}")

    # Full band structure from single diagonalization
    print("\n--- Band structure from Fourier eigenstates ---")
    L = 50
    k = 0.0  # any k works; each ℓ gives p_eff = k + ℓθ
    H = build_fourier_matrix(k, L, a_hat, b_hat)
    evals, evecs = np.linalg.eig(H)
    dim = 4 * (2*L + 1)

    # For each eigenstate, find the dominant Fourier mode ℓ
    th = THETA_BC
    results = []  # (p_eff, phase, dominant_ℓ, concentration)
    for i in range(dim):
        v = evecs[:, i]
        # Compute weight in each ℓ block
        weights = np.zeros(2*L + 1)
        for il in range(2*L + 1):
            block = v[4*il:4*il+4]
            weights[il] = np.sum(np.abs(block)**2)
        weights /= np.sum(weights)

        dom_il = np.argmax(weights)
        dom_ell = dom_il - L
        concentration = weights[dom_il]
        p_eff = (k + dom_ell * th) % (2 * np.pi)
        if p_eff > np.pi:
            p_eff -= 2 * np.pi

        results.append((p_eff, np.angle(evals[i]), dom_ell, concentration))

    results.sort(key=lambda x: x[0])

    print(f"L={L}, k={k}, {dim} eigenvalues")
    print(f"\nSample (every 10th):")
    print(f"{'p_eff/π':>8s}  {'E/π':>10s}  {'ℓ':>4s}  {'conc':>6s}")
    for idx in range(0, len(results), 10):
        r = results[idx]
        print(f"{r[0]/np.pi:8.4f}  {r[1]/np.pi:10.6f}  {r[2]:4d}  {r[3]:6.4f}")

    # Compare with finite chain spectrum
    print("\n--- Comparison with finite chain ---")
    from src.walk import build_shift_operator, build_helix_taus

    N_chain = 400
    print(f"Building finite chain S (N={N_chain})...")
    t0 = time.time()
    tau_list = build_helix_taus(N_chain)
    S = build_shift_operator(N_chain, tau_list)
    chain_evals = np.linalg.eigvals(S)
    chain_phases = np.sort(np.angle(chain_evals))
    print(f"  Done in {time.time()-t0:.1f}s, {len(chain_phases)} eigenvalues")

    # Fourier band structure
    qb_phases = np.array([r[1] for r in results])
    qb_peff = np.array([r[0] for r in results])

    # For each finite-chain eigenvalue, find closest quasi-Bloch eigenvalue
    errors = []
    for cp in chain_phases:
        diffs = np.abs(qb_phases - cp)
        # Also check wrapping
        diffs2 = np.abs(qb_phases - cp + 2*np.pi)
        diffs3 = np.abs(qb_phases - cp - 2*np.pi)
        min_diff = min(np.min(diffs), np.min(diffs2), np.min(diffs3))
        errors.append(min_diff)
    errors = np.array(errors)

    print(f"\nMatching finite chain eigenvalues to quasi-Bloch eigenvalues:")
    print(f"  Mean error:   {np.mean(errors)/np.pi:.6f} π")
    print(f"  Max error:    {np.max(errors)/np.pi:.6f} π")
    print(f"  Median error: {np.median(errors)/np.pi:.6f} π")

    # Save data for plotting
    np.savez("scripts/quasi_bloch_bands.npz",
             p_eff=qb_peff, phases=qb_phases,
             chain_phases=chain_phases,
             ell_values=np.array([r[2] for r in results]),
             concentrations=np.array([r[3] for r in results]))
    print("\nData saved to scripts/quasi_bloch_bands.npz")


if __name__ == "__main__":
    main()
