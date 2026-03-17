# Quantum Random Walks on a Tetrahedral Lattice

Constructing a discrete-time quantum random walk (DTQW) in 3+1 dimensions that converges to the Dirac equation in the continuum limit, where walker steps form a regular tetrahedron.

## Approach

We construct operators τ₁, τ₂, τ₃, τ₄ (4×4 matrices, each unitarily equivalent to diag(-1,-1,1,1)) satisfying:

```
Σₐ τₐ Dₐ = α₁ ∂ₓ + α₂ ∂ᵧ + α₃ ∂_z
```

where Dₐ are directional derivatives along tetrahedral directions and αᵢ are the Dirac alpha matrices. This rewrites the Dirac Hamiltonian in tetrahedral coordinates.

The core open problem is constructing a **unitary** time-evolution operator from these τ-based shift operators.

## Structure

- `src/` — Python modules (Dirac algebra, tetrahedral geometry, τ operators, walk operators)
- `notebooks/` — Jupyter notebooks documenting derivations
- `tests/` — pytest test suite

## Setup

```bash
pip install -r requirements.txt
pytest
```
