---
name: Quasi-Bloch decomposition results
description: Walk eigenstates decompose via e^{ikn}e^{imnθ}; τ has only l=0,±1; W(k) requires e^{∓imθ} shift phase; P±-sym IC best for Dirac comparison
type: project
---

## Quasi-Bloch framework
- Walk eigenstates: ψ(n) = e^{ikn} Σ_m c_m e^{imnθ}, θ = arccos(-2/3)
- τ(φ) has exactly l = 0, ±1 Fourier harmonics (99.5% in 2 harmonics)
- Combined walk blocks F̂, Ĝ: ||F̂₀|| = 1.33, ||F̂±₁|| = 0.34, |l|≥2 ≤ 0.014

## Critical phase factor in W(k)
**[W(k)]_{m',m} = e^{-ik} e^{-imθ} F̂_{m'-m} + e^{ik} e^{+imθ} Ĝ_{m'-m}**

The e^{∓imθ} comes from the shift u(n∓1) in the Bloch basis. Without it:
- Eigenvector weight was 99.96% in m=3 (wrong!) → high-freq IC oscillations
- Band structure offset from Dirac prediction
With correction: m=0 has 93.9% weight (correct), band structure matches Dirac.

## Eigenstate analysis
- Bulk eigenstates: F ≈ constant (R² ≈ 0.01), nearly pure plane waves
- Band-edge states: F develops smooth sinusoidal φ-dependence (R² → 1)
- Walk eigenstate spinor at k=0: (0.11, 0.87, 0.43, 0.20) — differs from Dirac β-symmetric (0.71, 0, 0.71, 0)

## IC comparison (confirmed with corrected phase)
- **P±-sym transported: 3.7% RMS, 1.8% asymmetry** — best for Dirac comparison
- Walk m=0 spinor transported: 11.4% RMS — different physical state
- Quasi-Bloch dressed: poor — eigenstates don't match β-symmetric Dirac IC
- P±-sym works because it matches the physical setup (equal P+/P-), not any eigenstate

## Scripts
- scripts/quasi_bloch.py — eigenstate decomposition
- scripts/quasi_bloch_operator.py — Fourier analysis
- scripts/quasi_bloch_truncated.py — truncated operator (NEEDS phase fix)
- scripts/quasi_bloch_ic.py — IC comparison (NEEDS phase fix)
- scripts/quasi_bloch_l1.py — L1/RMS comparison
- scripts/dispersion_publication.py — publication figure (phase FIXED)
