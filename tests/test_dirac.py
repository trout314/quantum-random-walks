"""Tests for Dirac algebra infrastructure."""

from src.dirac import (
    verify_clifford_algebra, verify_alpha_properties,
    alpha_sym, beta_sym, gamma_sym, gamma_5_sym, _I4_sym,
    spinor_rotation_generator,
)
from sympy import zeros, simplify, I, eye


def test_clifford_algebra():
    results = verify_clifford_algebra()
    for (mu, nu), ok in results.items():
        assert ok, f"Clifford algebra failed for (mu={mu}, nu={nu})"


def test_alpha_properties():
    results = verify_alpha_properties()
    for name, ok in results.items():
        assert ok, f"Alpha property failed: {name}"


def test_gamma5_squared():
    """gamma_5^2 = I_4"""
    result = (gamma_5_sym * gamma_5_sym).applyfunc(simplify)
    assert result == _I4_sym


def test_gamma5_anticommutes_with_gamma():
    """Check {gamma_5, gamma^mu} = 0 for all mu."""
    for mu in range(4):
        anticomm = (gamma_5_sym * gamma_sym[mu] + gamma_sym[mu] * gamma_5_sym).applyfunc(simplify)
        assert anticomm == zeros(4), f"gamma_5 does not anticommute with gamma_{mu}"


def test_beta_squared():
    """beta^2 = I_4"""
    assert (beta_sym * beta_sym).applyfunc(simplify) == _I4_sym


def test_spinor_rotation_generators_antisymmetric():
    """S_{mu,nu} = -S_{nu,mu}"""
    for mu in range(4):
        for nu in range(mu + 1, 4):
            S_mn = spinor_rotation_generator(mu, nu)
            S_nm = spinor_rotation_generator(nu, mu)
            assert (S_mn + S_nm).applyfunc(simplify) == zeros(4)
