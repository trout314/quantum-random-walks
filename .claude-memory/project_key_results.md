---
name: project_key_results
description: Summary of all key mathematical results established so far in the quantum walk project
type: project
---

1. **τ operators:** τ_a = ±(√7/4)β + (3/4)(e_a·α) is the unique (in span{β,α_i}) family satisfying Dirac correspondence, involutory, Hermitian, eigenvalues {-1,-1,1,1}.

2. **Reflection rule:** Stepping through face a reflects directions via R_a(v) = v - 2(v·e_a)e_a. New site has -e_a as back direction. Coplanarity: e_i, e_a, w_i lie in a common plane.

3. **Anti-conjugation:** (e_s·α) τ_a (e_s·α) = -τ'_a (minus sign is intrinsic, no unitary conjugation gives +τ'_a).

4. **BC helix:** Face-stacking pattern [1,3,0,2] repeating = right-handed BC helix. Displacement -(2/3)e_a per step. Rotation arccos(-2/3) ≈ 131.81° per step (irrational). Perpendicular partner always [0,1,2,3] (hardcoded, verified).

5. **Perpendicular pairs:** 12 helix types through each point, forming 6 perpendicular L/R pairs. No mutually perpendicular TRIPLE exists.

6. **Disjoint spirals:** Same-pattern spirals from different sites are disjoint (zero overlap).

7. **Frame transport:** Polar decomposition of eigenspace overlap gives unitary U_{n,n+1} that intertwines projectors. Structure: identity on +1 eigenspace, SU(2) rotation on -1 eigenspace. Makes shift S unitary.

8. **Conveyor belt problem:** The intertwining frame transport creates a perfect conveyor belt: delta functions don't spread, momentum exp(ikn) has no effect, all eigenstates are spatially uniform. The transport erases all phase information between sites.

9. **Coin operator resolution:** Adding a site-local coin C_n = exp(-iθ e_{a(n)}·α) before the shift breaks the conveyor belt while preserving unitarity. The walk W = S·C produces:
   - **Massive Dirac dispersion:** E² = m² + v²k² with R² > 0.99
   - **Mass:** m ≈ 0.59 sin(θ), proportional to coin mixing amplitude
   - **Speed of light:** v ≈ 0.008 (constant across masses)
   - **Delta functions spread** (genuine dispersion)
   - **±E symmetry** preserved (particle-antiparticle)
   - Exactly unitary for all θ

10. **3D walk (W = S_R · S_L):** Isotropic BFS seed + chain expansion gives balanced lattice with 100% dual coverage. Stitched loops make both S_R and S_L unitary. Wavepacket shows isotropic 3D spreading. Coin operator not yet tested in 3D.

11. **Świerczkowski's theorem:** No closed chain of face-stacked tetrahedra exists (irrational rotation). Unique walker paths between any two locations. Open question: implications for two-slit interference.

**Current frontier:** The coin operator gives correct massive Dirac dispersion on the 1D helix. Next steps: (a) test the coin in 3D, (b) understand the factor 0.59 in m = 0.59 sin(θ), (c) understand the small velocity v ≈ 0.008.
