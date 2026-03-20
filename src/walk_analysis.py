"""
Read binary walk data from walk_gen and analyze the walk.
"""

import numpy as np
from scipy import sparse
import struct
import subprocess
import sys


def generate_lattice(chain_len=15, n_gen=1):
    """Run walk_gen and return the binary data."""
    proc = subprocess.run(
        ['./walk_gen', str(chain_len), str(n_gen)],
        capture_output=True
    )
    if proc.returncode != 0:
        print(proc.stderr.decode(), file=sys.stderr)
        raise RuntimeError("walk_gen failed")
    sys.stderr.write(proc.stderr.decode())
    return proc.stdout


def load_walk_data(data):
    """Parse binary walk data into positions and sparse matrices."""
    offset = 0

    # Header
    n_sites, n_interior, nnz_R, nnz_L = struct.unpack_from('4i', data, offset)
    offset += 16

    # Positions
    positions = np.frombuffer(data, dtype=np.float64, count=n_sites*3, offset=offset)
    positions = positions.reshape(n_sites, 3)
    offset += n_sites * 3 * 8

    # Membership
    membership = np.frombuffer(data, dtype=np.int32, count=n_sites*4, offset=offset)
    membership = membership.reshape(n_sites, 4)
    offset += n_sites * 4 * 4

    # S_R
    sr_rc = np.frombuffer(data, dtype=np.int32, count=nnz_R*2, offset=offset)
    sr_rc = sr_rc.reshape(nnz_R, 2)
    offset += nnz_R * 2 * 4
    sr_vals = np.frombuffer(data, dtype=np.float64, count=nnz_R*2, offset=offset)
    sr_vals = sr_vals.reshape(nnz_R, 2)
    offset += nnz_R * 2 * 8

    # S_L
    sl_rc = np.frombuffer(data, dtype=np.int32, count=nnz_L*2, offset=offset)
    sl_rc = sl_rc.reshape(nnz_L, 2)
    offset += nnz_L * 2 * 4
    sl_vals = np.frombuffer(data, dtype=np.float64, count=nnz_L*2, offset=offset)
    sl_vals = sl_vals.reshape(nnz_L, 2)
    offset += nnz_L * 2 * 8

    dim = 4 * n_sites
    S_R = sparse.csr_matrix(
        (sr_vals[:, 0] + 1j * sr_vals[:, 1], (sr_rc[:, 0], sr_rc[:, 1])),
        shape=(dim, dim)
    )
    S_L = sparse.csr_matrix(
        (sl_vals[:, 0] + 1j * sl_vals[:, 1], (sl_rc[:, 0], sl_rc[:, 1])),
        shape=(dim, dim)
    )

    return {
        'n_sites': n_sites,
        'n_interior': n_interior,
        'positions': positions,
        'membership': membership,
        'S_R': S_R,
        'S_L': S_L,
    }


def propagate_wavepacket(walk_data, sigma=1.5, n_steps=20, coin_state=None):
    """Initialize and propagate a Gaussian wavepacket."""
    N = walk_data['n_sites']
    positions = walk_data['positions']
    S_R = walk_data['S_R']
    S_L = walk_data['S_L']
    W = S_R @ S_L

    # Initialize Gaussian wavepacket at origin
    if coin_state is None:
        coin_state = np.array([1, 0, 0, 0], dtype=complex)
    coin_state = coin_state / np.linalg.norm(coin_state)

    psi = np.zeros(4 * N, dtype=complex)
    for i in range(N):
        r2 = np.dot(positions[i], positions[i])
        weight = np.exp(-r2 / (2 * sigma**2))
        psi[i*4:(i+1)*4] = weight * coin_state
    psi /= np.linalg.norm(psi)

    # Interior site mask
    mem = walk_data['membership']
    interior_mask = (mem[:, 0] >= 0) & (mem[:, 2] >= 0)  # has both R and L
    # Could further restrict to non-boundary, but this is a start

    results = []
    for t in range(n_steps + 1):
        prob = np.array([np.sum(np.abs(psi[i*4:(i+1)*4])**2) for i in range(N)])
        total_prob = np.sum(prob)
        norm = np.linalg.norm(psi)

        # Interior probability
        int_prob = np.sum(prob[interior_mask])

        # Position moments
        if total_prob > 1e-15:
            mean_pos = np.sum((prob / total_prob)[:, None] * positions, axis=0)
            mean_r2 = np.sum((prob / total_prob) * np.sum(positions**2, axis=1))
        else:
            mean_pos = np.zeros(3)
            mean_r2 = 0

        results.append({
            't': t,
            'norm': norm,
            'total_prob': total_prob,
            'int_prob': int_prob,
            'mean_pos': mean_pos,
            'mean_r2': mean_r2,
        })

        if t < n_steps:
            psi = W @ psi

    return results


if __name__ == '__main__':
    import time

    chain_len = 15
    n_gen = 1
    if len(sys.argv) > 1:
        chain_len = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_gen = int(sys.argv[2])

    print(f"Generating lattice (chain_len={chain_len}, n_gen={n_gen})...")
    t0 = time.time()
    data = generate_lattice(chain_len, n_gen)
    walk = load_walk_data(data)
    print(f"Loaded in {time.time()-t0:.2f}s: {walk['n_sites']} sites, "
          f"{walk['n_interior']} interior")
    print(f"S_R: {walk['S_R'].nnz} nnz, S_L: {walk['S_L'].nnz} nnz")

    # 3D check
    from numpy.linalg import svd
    centered = walk['positions'] - walk['positions'].mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)
    print(f"Position SV ratio: {S[2]/S[0]:.3f}")

    print(f"\nPropagating wavepacket...")
    t0 = time.time()
    results = propagate_wavepacket(walk, sigma=1.5, n_steps=15)
    print(f"Done in {time.time()-t0:.2f}s")

    print(f"\n{'t':>3} {'norm':>8} {'int_prob':>8} "
          f"{'<x>':>8} {'<y>':>8} {'<z>':>8} {'<r2>':>8}")
    for r in results:
        print(f"{r['t']:3d} {r['norm']:8.4f} {r['int_prob']:8.4f} "
              f"{r['mean_pos'][0]:8.4f} {r['mean_pos'][1]:8.4f} "
              f"{r['mean_pos'][2]:8.4f} {r['mean_r2']:8.4f}")
