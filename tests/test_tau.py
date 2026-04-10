"""Tests for τ operator construction."""

from src.tau_operators import construct_tau, verify_all


def test_tau_positive_sign():
    taus = construct_tau(sign=+1)
    results = verify_all(taus)

    for name, ok in results['dirac_correspondence'].items():
        assert ok, f"Dirac correspondence failed: {name}"

    for name, ok in results['involutory'].items():
        assert ok, f"Involutory check failed: {name}"

    for name, ok in results['hermitian'].items():
        assert ok, f"Hermiticity check failed: {name}"

    for name, eigs in results['eigenvalues'].items():
        assert eigs == {-1: 2, 1: 2}, f"Wrong eigenvalues for {name}: {eigs}"


def test_tau_negative_sign():
    taus = construct_tau(sign=-1)
    results = verify_all(taus)

    for name, ok in results['dirac_correspondence'].items():
        assert ok, f"Dirac correspondence failed: {name}"

    for name, ok in results['involutory'].items():
        assert ok, f"Involutory check failed: {name}"

    for name, ok in results['hermitian'].items():
        assert ok, f"Hermiticity check failed: {name}"

    for name, eigs in results['eigenvalues'].items():
        assert eigs == {-1: 2, 1: 2}, f"Wrong eigenvalues for {name}: {eigs}"
