"""Test D dirac module against Python reference implementation."""
import ctypes
import os
import numpy as np
import pytest
import sys

from src.dirac import alpha as ALPHA, beta as BETA, I4
from src.walk import make_tau_from_dir, frame_transport as py_frame_transport
from src.tetrahedron import vertices

_dp = ctypes.POINTER(ctypes.c_double)

def _load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, '..', 'dlang', 'build', 'quantum_walk.so')
    if not os.path.exists(so_path):
        pytest.skip("quantum_walk.so not built")
    lib = ctypes.CDLL(so_path)
    lib.make_tau_c.restype = None
    lib.make_tau_c.argtypes = [_dp, _dp, _dp]
    lib.frame_transport_c.restype = None
    lib.frame_transport_c.argtypes = [_dp, _dp, _dp, _dp, _dp, _dp]
    return lib

_lib = _load_lib()


def d_make_tau(direction):
    """Call D make_tau_c and return a 4x4 complex numpy array."""
    d = np.array(direction, dtype=np.float64)
    re = np.zeros(16, dtype=np.float64)
    im = np.zeros(16, dtype=np.float64)
    _lib.make_tau_c(d.ctypes.data_as(_dp), re.ctypes.data_as(_dp), im.ctypes.data_as(_dp))
    return (re + 1j * im).reshape(4, 4)


def d_frame_transport(tau_from, tau_to):
    """Call D frame_transport_c and return a 4x4 complex numpy array."""
    tf_re = np.ascontiguousarray(tau_from.real.ravel(), dtype=np.float64)
    tf_im = np.ascontiguousarray(tau_from.imag.ravel(), dtype=np.float64)
    tt_re = np.ascontiguousarray(tau_to.real.ravel(), dtype=np.float64)
    tt_im = np.ascontiguousarray(tau_to.imag.ravel(), dtype=np.float64)
    out_re = np.zeros(16, dtype=np.float64)
    out_im = np.zeros(16, dtype=np.float64)
    _lib.frame_transport_c(
        tf_re.ctypes.data_as(_dp), tf_im.ctypes.data_as(_dp),
        tt_re.ctypes.data_as(_dp), tt_im.ctypes.data_as(_dp),
        out_re.ctypes.data_as(_dp), out_im.ctypes.data_as(_dp),
    )
    return (out_re + 1j * out_im).reshape(4, 4)


def test_tau_matches_python():
    """D τ operators match Python reference for all 4 tet directions."""
    for a in range(4):
        d = np.array([float(v) for v in vertices[a]])
        d_tau = d_make_tau(d)
        py_tau = make_tau_from_dir(d)
        np.testing.assert_allclose(d_tau, py_tau, atol=1e-14,
                                   err_msg=f"τ mismatch for direction {a}")


def test_tau_involutory():
    """D τ² = I for all 4 directions."""
    for a in range(4):
        d = np.array([float(v) for v in vertices[a]])
        tau = d_make_tau(d)
        np.testing.assert_allclose(tau @ tau, np.eye(4), atol=1e-13)


def test_frame_transport_matches_python():
    """D frame transport matches Python reference."""
    dirs = [np.array([float(v) for v in vertices[a]]) for a in range(4)]
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            tau_i = make_tau_from_dir(dirs[i])
            tau_j = make_tau_from_dir(dirs[j])
            d_U = d_frame_transport(tau_i, tau_j)
            py_U = py_frame_transport(tau_i, tau_j)
            np.testing.assert_allclose(d_U, py_U, atol=1e-12,
                                       err_msg=f"Frame transport mismatch {i}->{j}")


def test_frame_transport_unitary():
    """D frame transport U is unitary: U†U = I."""
    dirs = [np.array([float(v) for v in vertices[a]]) for a in range(4)]
    tau0 = make_tau_from_dir(dirs[0])
    tau1 = make_tau_from_dir(dirs[1])
    U = d_frame_transport(tau0, tau1)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(4), atol=1e-12)


def test_frame_transport_intertwines():
    """D frame transport intertwines: U τ_from = τ_to U."""
    dirs = [np.array([float(v) for v in vertices[a]]) for a in range(4)]
    tau0 = make_tau_from_dir(dirs[0])
    tau2 = make_tau_from_dir(dirs[2])
    U = d_frame_transport(tau0, tau2)
    np.testing.assert_allclose(U @ tau0, tau2 @ U, atol=1e-12)
