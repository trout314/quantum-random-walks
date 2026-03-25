"""Test D/Python interop via ctypes."""
import ctypes
import os
import numpy as np
import pytest

_dp = ctypes.POINTER(ctypes.c_double)

def _load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, '..', 'dlang', 'build', 'quantum_walk.so')
    if not os.path.exists(so_path):
        pytest.skip("quantum_walk.so not built — run: ninja -C dlang/build")
    lib = ctypes.CDLL(so_path)

    lib.walk_version.restype = ctypes.c_char_p
    lib.walk_version.argtypes = []

    lib.scale_array.restype = ctypes.c_int
    lib.scale_array.argtypes = [_dp, ctypes.c_int, ctypes.c_double]

    lib.dot_product.restype = ctypes.c_int
    lib.dot_product.argtypes = [_dp, _dp, ctypes.c_int, _dp]

    lib.apply_matrix_2x2.restype = ctypes.c_int
    lib.apply_matrix_2x2.argtypes = [
        _dp, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
    ]

    return lib


_lib = _load_lib()


def test_version():
    v = _lib.walk_version()
    assert v == b"quantum-walk-d 0.1.0"


def test_scale_array():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ret = _lib.scale_array(a.ctypes.data_as(_dp), len(a), 2.5)
    assert ret == 0
    np.testing.assert_allclose(a, [2.5, 5.0, 7.5, 10.0])


def test_scale_array_in_place():
    """Verify D modifies the numpy array's memory directly."""
    a = np.array([10.0, 20.0], dtype=np.float64)
    addr_before = a.ctypes.data
    _lib.scale_array(a.ctypes.data_as(_dp), len(a), 0.1)
    assert a.ctypes.data == addr_before  # same memory
    np.testing.assert_allclose(a, [1.0, 2.0])


def test_dot_product():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    result = ctypes.c_double(0.0)
    ret = _lib.dot_product(
        a.ctypes.data_as(_dp),
        b.ctypes.data_as(_dp),
        len(a),
        ctypes.byref(result),
    )
    assert ret == 0
    assert result.value == pytest.approx(32.0)  # 4+10+18


def test_dot_product_large():
    n = 100000
    a = np.ones(n, dtype=np.float64)
    b = np.arange(n, dtype=np.float64)
    result = ctypes.c_double(0.0)
    ret = _lib.dot_product(
        a.ctypes.data_as(_dp),
        b.ctypes.data_as(_dp),
        n,
        ctypes.byref(result),
    )
    assert ret == 0
    assert result.value == pytest.approx(n * (n - 1) / 2)


def test_apply_matrix_2x2():
    # Rotation by 90 degrees: [[0,-1],[1,0]]
    data = np.array([1.0, 0.0, 0.0, 1.0, 3.0, 4.0], dtype=np.float64)
    ret = _lib.apply_matrix_2x2(data.ctypes.data_as(_dp), 3, 0, -1, 1, 0)
    assert ret == 0
    np.testing.assert_allclose(data, [0.0, 1.0, -1.0, 0.0, -4.0, 3.0])


def test_error_on_null():
    ret = _lib.scale_array(None, 5, 1.0)
    assert ret == -1


def test_error_on_zero_length():
    a = np.array([1.0], dtype=np.float64)
    ret = _lib.scale_array(a.ctypes.data_as(_dp), 0, 1.0)
    assert ret == -1
