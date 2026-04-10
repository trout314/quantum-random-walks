#!/usr/bin/env python3
"""
Visualize BFS growth shells on the 3D tetrahedral lattice.

Builds the lattice from perpendicular R/L BC helix chains, defines
neighbors as chain-adjacent sites (±1 along R or L chain), then
does BFS from the center and plots 5 panels showing successive shells.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque, defaultdict
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from lattice3d import Lattice3D


def build_neighbor_graph(lat):
    """
    Build adjacency list from chain connectivity.
    Two sites are neighbors if they are adjacent along any shared chain.
    """
    n = len(lat.sites)
    neighbors = defaultdict(set)

    # Group sites by chain
    r_chains = defaultdict(list)  # chain_id -> [(r_idx, site_id)]
    l_chains = defaultdict(list)  # chain_id -> [(l_idx, site_id)]

    for s in lat.sites:
        if s.r_chain_id >= 0:
            r_chains[s.r_chain_id].append((s.r_idx, s.site_id))
        if s.l_chain_id >= 0:
            l_chains[s.l_chain_id].append((s.l_idx, s.site_id))

    # Connect adjacent sites along each chain
    for chain_sites in list(r_chains.values()) + list(l_chains.values()):
        chain_sites.sort()  # sort by chain index
        for i in range(len(chain_sites) - 1):
            idx_a, sid_a = chain_sites[i]
            idx_b, sid_b = chain_sites[i + 1]
            if idx_b - idx_a == 1:  # truly adjacent
                neighbors[sid_a].add(sid_b)
                neighbors[sid_b].add(sid_a)

    return neighbors


def bfs_distances(neighbors, start, n_sites):
    dist = np.full(n_sites, -1, dtype=int)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def main():
    r_extent = 8
    l_extent = 8
    depth = 1

    print("Building 3D lattice...")
    lat = Lattice3D()
    lat.build_from_seed(r_extent=r_extent, l_extent=l_extent, depth=depth)
    n_sites = len(lat.sites)
    print(f"  {n_sites} sites, {len(lat.chains)} chains")

    print("Building neighbor graph...")
    neighbors = build_neighbor_graph(lat)

    # Find the seed site (r_chain=0, r_idx=0)
    seed = None
    for s in lat.sites:
        if s.r_chain_id == 0 and s.r_idx == 0:
            seed = s.site_id
            break
    if seed is None:
        seed = 0
    print(f"Seed site: {seed}")

    print("Running BFS...")
    dist = bfs_distances(neighbors, seed, n_sites)

    max_dist = dist[dist >= 0].max()
    print(f"  Max BFS distance: {max_dist}")
    for d in range(min(8, max_dist + 1)):
        count = np.sum(dist == d)
        print(f"  d={d}: {count} sites")

    # Gather positions
    positions = np.array([s.pos for s in lat.sites])

    # Center on seed
    origin = positions[seed]
    positions = positions - origin

    # Determine consistent view angle
    # Find the extent of the first 5 shells
    max_shell = 5
    mask_all = (dist >= 0) & (dist <= max_shell)
    pos_vis = positions[mask_all]
    extent = np.max(np.abs(pos_vis)) * 1.15 if len(pos_vis) > 0 else 5

    # Colors for BFS distance
    shell_colors = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                    '#a65628', '#f781bf', '#999999']

    # ---- Figure: 5 panels, each showing cumulative BFS growth ----
    fig = plt.figure(figsize=(28, 7), facecolor='white')

    # Tight extent: fit to the sites we'll actually show
    max_shell = 5
    mask_shown = (dist >= 0) & (dist <= max_shell)
    if np.any(mask_shown):
        pos_shown = positions[mask_shown]
        extent = np.max(np.abs(pos_shown)) * 1.15
    else:
        extent = 3

    elev, azim = 22, 35

    for panel in range(5):
        ax = fig.add_subplot(1, 5, panel + 1, projection='3d')
        ax.set_facecolor('white')
        d_max = panel + 1  # show shells 0..d_max

        # Plot all sites up to this distance, newest shell on top
        for d in range(d_max + 1):
            mask = dist == d
            if not np.any(mask):
                continue
            p = positions[mask]
            color = shell_colors[d % len(shell_colors)]
            # Seed bigger, older shells slightly faded, newest shell bold
            if d == 0:
                size, alpha = 80, 1.0
            elif d == d_max:
                size, alpha = 50, 1.0
            else:
                size, alpha = 30, 0.5
            ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                       s=size, c=color, alpha=alpha,
                       edgecolors='none', depthshade=False,
                       label=f'd={d} ({np.sum(mask)})' if panel == 4 else None,
                       zorder=10 + d)

        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        # Clean: no ticks, no axis labels, no panes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.zaxis.set_ticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.grid(True, alpha=0.15)
        n_cumul = np.sum((dist >= 0) & (dist <= d_max))
        ax.set_title(f'BFS ≤ {d_max}\n{n_cumul} sites', fontsize=11, pad=0)
        ax.view_init(elev=elev, azim=azim)

    # Legend on last panel
    fig.axes[-1].legend(fontsize=8, loc='upper left', markerscale=1.5,
                        framealpha=0.8)

    fig.suptitle(
        'BFS growth on 3D tetrahedral lattice',
        fontsize=14, y=0.98)
    plt.tight_layout()

    out = '/tmp/bfs_3d_growth.png'
    plt.savefig(out, dpi=150)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
