# Key Results: Quantum Walk on BC Helix Chain

## Continuum Limit (1D)

The walk with the **dual parity coin** (f₁·α, f₂·α perpendicular to face direction e)
has a proper continuum limit that reproduces the **massive 1D Dirac equation**.

### Scaling law

The group velocity v depends on the dimensionless product C = θ·σ:

```
v(C) = c / √(c² + C²)
```

where c ≈ 0.619 is the speed of light (in lattice units), θ is the coin angle,
and σ is the Gaussian width (wavelength scale).

### Evidence

| C = θ·σ | v (σ=200) | v (σ=500) | v (σ=1000) | v (σ=2000) | CV |
|---------|-----------|-----------|------------|------------|-----|
| 1 | 0.4748 | 0.4746 | 0.4746 | 0.4746 | 0.0% |
| 2 | 0.3272 | 0.3267 | 0.3265 | 0.3264 | 0.1% |
| 5 | 0.1900 | 0.1880 | 0.1875 | 0.1872 | 0.6% |
| 10 | 0.1363 | 0.1226 | 0.1201 | 0.1193 | 5.5% |

### Interpretation

- **Mass**: m = θ in lattice units (coin angle IS the mass parameter)
- **Speed of light**: c ≈ 0.619 lattice spacings per time step
- **Massless limit** (θ=0): walk exactly reproduces massless Dirac equation
- **Massive limit** (large θ·σ): v → 0, particle at rest
- **Lattice corrections**: grow with C, ~5% at C=10

### Walk operator

W = S · C₂ · C₁ where:
- S: shift along BC helix chain with tau-based P+/P- projectors and frame transport
- C₁ = cos(θ)I - i sin(θ)(f₁·α) with f₁ ⊥ e
- C₂ = cos(θ)I - i sin(θ)(f₂·α) with f₂ = e × f₁
- τ = (√7/4)β + (3/4)(e·α) at each site
