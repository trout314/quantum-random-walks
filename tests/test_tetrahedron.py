"""Tests for tetrahedral geometry."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sympy import simplify, Rational, Matrix

from src.tetrahedron import (
    verify_unit_vectors, verify_centroid, verify_dot_products,
    verify_isotropy, isotropy_tensor, vertices,
    directions_after_path, position_after_path, reflect_directions,
    bc_helix_path, BC_HELIX_R, BC_HELIX_L,
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


def _dirs_equal(d1, d2):
    return all(
        (a - b).applyfunc(simplify) == Matrix([0, 0, 0])
        for a, b in zip(d1, d2)
    )


def test_empty_path():
    assert _dirs_equal(directions_after_path([]), vertices)


def test_step_and_back():
    """Stepping out and back recovers original tetrahedron."""
    for a in range(4):
        assert _dirs_equal(directions_after_path([a, a]), vertices)


def test_reflected_directions_form_tetrahedron():
    """After any single step, new directions form a regular tetrahedron."""
    for a in range(4):
        dirs = directions_after_path([a])
        # Unit vectors
        for d in dirs:
            assert simplify(d.dot(d) - 1) == 0
        # Dot products -1/3
        for i in range(4):
            for j in range(i + 1, 4):
                assert simplify(dirs[i].dot(dirs[j])) == Rational(-1, 3)


def test_step_reverses_direction():
    """After stepping in e_a, the a-th direction at new site is -e_a."""
    for a in range(4):
        dirs = directions_after_path([a])
        assert (dirs[a] + vertices[a]).applyfunc(simplify) == Matrix([0, 0, 0])


def test_coplanarity():
    """e_i, e_step, w_i are coplanar for i != step."""
    for step in range(4):
        dirs = directions_after_path([step])
        for i in range(4):
            if i == step:
                continue
            M = vertices[i].row_join(vertices[step]).row_join(dirs[i])
            assert simplify(M.det()) == 0


def test_position_tracking():
    pos, dirs = position_after_path([0])
    assert pos.applyfunc(simplify) == vertices[0]

    pos, dirs = position_after_path([0, 0])
    assert pos.applyfunc(simplify) == Matrix([0, 0, 0])


def test_bc_helix_step_size():
    """BC helix steps have constant magnitude 2/3."""
    positions, _ = bc_helix_path(8, chirality='R')
    for i in range(8):
        step = (positions[i + 1] - positions[i]).applyfunc(simplify)
        mag_sq = simplify(step.dot(step))
        assert mag_sq == Rational(4, 9), f"Step {i} magnitude^2 = {mag_sq}"


def test_bc_helix_directions_valid():
    """Directions at each BC helix site form a regular tetrahedron."""
    dirs = list(vertices)
    pattern = BC_HELIX_R
    for step in range(8):
        idx = pattern[step % 4]
        dirs = [(d - 2 * d.dot(dirs[idx]) * dirs[idx]).applyfunc(simplify)
                for d in dirs]
        for i in range(4):
            assert simplify(dirs[i].dot(dirs[i]) - 1) == 0
        for i in range(4):
            for j in range(i + 1, 4):
                assert simplify(dirs[i].dot(dirs[j])) == Rational(-1, 3)
