"""Tests for tetrahedral geometry."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tetrahedron import (
    verify_unit_vectors, verify_centroid, verify_dot_products,
    verify_isotropy, isotropy_tensor, vertices,
)


def test_unit_vectors():
    results = verify_unit_vectors()
    for name, ok in results.items():
        assert ok, f"Failed: {name}"


def test_centroid_at_origin():
    assert verify_centroid(), "Vertices do not sum to zero"


def test_dot_products():
    results = verify_dot_products()
    for name, ok in results.items():
        assert ok, f"Failed: {name}"


def test_isotropy():
    assert verify_isotropy(), "Isotropy tensor is not (4/3) delta_{ij}"
