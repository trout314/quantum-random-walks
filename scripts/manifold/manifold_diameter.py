#!/usr/bin/env python3
"""Measure graph diameter of manifold triangulations across sizes."""
import numpy as np
import os, sys, glob
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from triangulation import Triangulation

SEEDS_DIR = os.path.expanduser('~/Desktop/Discrete-Differential-Geometry/seeds')
SIZES = ['1e2', '178', '316', '562', '1778', '3162', '5623', '17783', '31623', '56234']


def find_seed(target_n):
    pattern = os.path.join(SEEDS_DIR, f'S3_N{target_n}_1e-1_ED5p0043_1e-1_s000.mfd')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    pattern = os.path.join(SEEDS_DIR, f'S3_N{target_n}_*_s000.mfd')
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def bfs_eccentricity(neighbors, start):
    """BFS from start, return max distance."""
    n = len(neighbors)
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])
    max_d = 0
    while queue:
        t = queue.popleft()
        for nb in neighbors[t]:
            if nb >= 0 and dist[nb] < 0:
                dist[nb] = dist[t] + 1
                if dist[nb] > max_d:
                    max_d = dist[nb]
                queue.append(nb)
    return max_d


def estimate_diameter(tri, n_samples=20):
    """Estimate diameter by BFS from random tets."""
    rng = np.random.default_rng(42)
    samples = rng.choice(tri.n_tets, min(n_samples, tri.n_tets), replace=False)
    max_ecc = 0
    for s in samples:
        ecc = bfs_eccentricity(tri.neighbors, s)
        if ecc > max_ecc:
            max_ecc = ecc
    return max_ecc


def main():
    print(f"{'tets':>8} {'verts':>7} {'diam':>6} {'N^(1/3)':>8} {'diam/N^(1/3)':>12}")
    print('-' * 50)

    for size in SIZES:
        path = find_seed(size)
        if path is None:
            print(f"  No seed for N={size}")
            continue
        tri = Triangulation.load(path)
        diam = estimate_diameter(tri)
        n13 = tri.n_tets ** (1/3)
        print(f"{tri.n_tets:8d} {tri.n_verts:7d} {diam:6d} {n13:8.2f} {diam/n13:12.2f}")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
