"""Test D geometry module against Python reference implementation."""
import ctypes
import os
import numpy as np
import pytest
import sys

from src.tetrahedron import vertices, reflect_directions, position_after_path

_dp = ctypes.POINTER(ctypes.c_double)

def _load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, '..', 'dlang', 'build', 'quantum_walk.so')
    if not os.path.exists(so_path):
        pytest.skip("quantum_walk.so not built")
    lib = ctypes.CDLL(so_path)
    lib.tet_dirs.restype = None
    lib.tet_dirs.argtypes = [_dp]
    lib.helix_step_c.restype = None
    lib.helix_step_c.argtypes = [_dp, _dp, ctypes.c_int]
    return lib

_lib = _load_lib()


def test_tet_dirs_match_python():
    """D tetrahedral directions match the Python reference."""
    out = np.zeros(12, dtype=np.float64)
    _lib.tet_dirs(out.ctypes.data_as(_dp))
    d_dirs = out.reshape(4, 3)

    py_dirs = np.array([[float(v) for v in vert] for vert in vertices])

    np.testing.assert_allclose(d_dirs, py_dirs, atol=1e-14)


def test_helix_step_matches_python():
    """D helix step matches Python reflect_directions for all 4 faces."""
    py_dirs_sympy = list(vertices)

    for face in range(4):
        # D version: one helix step
        d_pos = np.zeros(3, dtype=np.float64)
        py_dirs_float = np.array([[float(v) for v in vert] for vert in py_dirs_sympy])
        d_dirs = py_dirs_float.flatten().copy()
        _lib.helix_step_c(
            d_pos.ctypes.data_as(_dp),
            d_dirs.ctypes.data_as(_dp),
            face,
        )
        d_dirs = d_dirs.reshape(4, 3)

        # Python version: reflect_directions gives new dirs
        py_new_dirs = reflect_directions(py_dirs_sympy, face)
        py_new_dirs_float = np.array([[float(v) for v in d] for d in py_new_dirs])

        # Python position: step = -2/3 * d[face]
        step = np.array([float(v) for v in py_dirs_sympy[face]]) * (-2.0 / 3.0)

        np.testing.assert_allclose(d_pos, step, atol=1e-14,
                                   err_msg=f"Position mismatch for face {face}")
        np.testing.assert_allclose(d_dirs, py_new_dirs_float, atol=1e-14,
                                   err_msg=f"Directions mismatch for face {face}")


def test_helix_path_self_consistency():
    """Multi-step D helix path: step sizes are always 2/3, directions stay valid."""
    path = [1, 3, 0, 2] * 5  # 20 steps

    pos = np.zeros(3, dtype=np.float64)
    py_dirs_float = np.array([[float(v) for v in vert] for vert in vertices])
    dirs = py_dirs_float.flatten().copy()

    prev_pos = pos.copy()
    for face in path:
        _lib.helix_step_c(
            pos.ctypes.data_as(_dp),
            dirs.ctypes.data_as(_dp),
            face,
        )
        # Each step moves exactly 2/3
        step = np.linalg.norm(pos - prev_pos)
        assert abs(step - 2.0 / 3.0) < 1e-13, f"Step size {step} != 2/3"
        prev_pos = pos.copy()

        # Directions remain unit vectors
        d = dirs.reshape(4, 3)
        for i in range(4):
            assert abs(np.linalg.norm(d[i]) - 1.0) < 1e-12

        # Directions still sum to zero
        assert np.linalg.norm(d.sum(axis=0)) < 1e-12
