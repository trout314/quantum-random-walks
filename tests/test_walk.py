"""Tests for walk operators."""

import numpy as np
from src.walk import (
    build_shift_operator, build_helix_taus, analyze_spectrum,
    frame_transport, make_tau, vertices_numpy, I4,
)


def test_tau_involutory():
    """Each τ operator satisfies τ² = I."""
    taus = build_helix_taus(8)
    for n, tau in enumerate(taus):
        err = np.linalg.norm(tau @ tau - I4)
        assert err < 1e-12, f"tau_{n}^2 != I, error = {err}"


def test_frame_transport_unitary():
    """Frame transport is unitary."""
    taus = build_helix_taus(8)
    for n in range(7):
        U = frame_transport(taus[n], taus[n + 1])
        err = np.linalg.norm(U @ U.conj().T - I4)
        assert err < 1e-12, f"U_{n} not unitary, error = {err}"


def test_frame_transport_intertwines():
    """Frame transport maps eigenspaces correctly."""
    taus = build_helix_taus(8)
    for n in range(7):
        U = frame_transport(taus[n], taus[n + 1])
        mapped = U @ taus[n] @ U.conj().T
        err = np.linalg.norm(mapped - taus[n + 1])
        assert err < 1e-10, f"U_{n} doesn't intertwine, error = {err}"


def test_shift_unitary_small():
    """Shift operator is unitary for small chain."""
    N = 8
    S = build_shift_operator(N)
    err = np.linalg.norm(S.conj().T @ S - np.eye(4 * N))
    assert err < 1e-10, f"S not unitary, error = {err}"


def test_shift_unitary_medium():
    """Shift operator is unitary for medium chain."""
    N = 40
    S = build_shift_operator(N)
    err = np.linalg.norm(S.conj().T @ S - np.eye(4 * N))
    assert err < 1e-6, f"S not unitary, error = {err}"


def test_eigenvalues_on_unit_circle():
    """All eigenvalues lie on the unit circle."""
    N = 40
    S = build_shift_operator(N)
    evals = np.linalg.eigvals(S)
    max_dev = max(abs(abs(evals) - 1))
    assert max_dev < 1e-6, f"Eigenvalue off unit circle by {max_dev}"


def test_spectrum_pm_symmetric():
    """Spectrum has ±E symmetry."""
    N = 40
    S = build_shift_operator(N)
    result = analyze_spectrum(S, N)
    phases = np.sort(result['phases'])
    # Check that for each phase p, -p also appears
    for p in phases:
        matches = [q for q in phases if abs(q + p) < 0.001]
        assert len(matches) >= 1, f"No partner for phase {p:.4f}"


def test_linear_dispersion():
    """Bands have linear dispersion E = E₀ + v·k with v = 2π/N."""
    N = 40
    S = build_shift_operator(N)
    result = analyze_spectrum(S, N)
    delta_k = 2 * np.pi / N

    for i, band in enumerate(result['bands']):
        if len(band) < 3:
            continue
        spacings = np.diff(band)
        # All spacings should equal delta_k
        for j, s in enumerate(spacings):
            assert abs(s - delta_k) < 0.01, \
                f"Band {i}, spacing {j}: {s:.6f} != {delta_k:.6f}"
