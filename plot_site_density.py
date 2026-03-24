#!/usr/bin/env python3
"""Plot radial site density for different generation modes."""

import numpy as np
import matplotlib.pyplot as plt

configs = [
    ('/tmp/sites_bfs_20.dat', 'BFS, max_per_cell=20 (current)'),
    ('/tmp/sites_bfs_8.dat', 'BFS, max_per_cell=8'),
    ('/tmp/sites_chain_8.dat', 'Chain-only, max_per_cell=8'),
]

# Tetrahedron volume (edge length = 1)
V_tet = 1.0 / (6 * np.sqrt(2))  # ~0.1178

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for fname, label in configs:
    data = np.loadtxt(fname)
    r = data[:, 3]

    # Radial density: count sites in spherical shells
    dr = 0.5
    r_max = np.max(r)
    bins = np.arange(0, r_max + dr, dr)
    counts, _ = np.histogram(r, bins=bins)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    # Volume of each shell: 4π r² dr
    shell_vol = 4 * np.pi * r_mid**2 * dr
    # Avoid division by zero at r=0
    shell_vol[shell_vol < 1e-10] = 1e-10
    density = counts / shell_vol  # sites per unit volume

    nsites = len(r)
    axes[0].plot(r_mid, density, linewidth=1.2, label=f'{label} (N={nsites})')
    axes[1].plot(r_mid, density * V_tet, linewidth=1.2, label=f'{label}')

# Target line: 1 site per tet volume
axes[0].axhline(y=1.0/V_tet, color='k', linestyle=':', linewidth=1, label=f'1/V_tet = {1/V_tet:.1f}')
axes[0].set_title('Radial site density — σ=5.0')
axes[0].set_xlabel('r')
axes[0].set_ylabel('Sites per unit volume')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, 25)
axes[0].set_ylim(0, None)

axes[1].axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='Target: 1 site per V_tet')
axes[1].set_title('Sites per tetrahedron volume')
axes[1].set_xlabel('r')
axes[1].set_ylabel('Sites × V_tet')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, 25)
axes[1].set_ylim(0, None)

plt.tight_layout()
out = '/tmp/site_radial_density.png'
plt.savefig(out, dpi=150)
print(f'Graph saved to: {out}')
