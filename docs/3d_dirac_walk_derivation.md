# 3D Dirac Equation as a Quantum Walk: Derivation

## The 3D Dirac Equation in Tetrahedral Coordinates

The Dirac Hamiltonian, rewritten using the τ operators:

    H = Σ_a τ_a D_a + m β

where D_a = e_a · ∇ is the directional derivative along tetrahedral
direction e_a, and the Dirac correspondence Σ_a e_a^i τ_a = α_i ensures
this equals the standard form H = Σ_i α_i p_i + m β.

Each τ_a = (√7/4)β + (3/4)(e_a · α) is Hermitian and involutory (τ_a² = I),
with eigenvalues {-1, -1, 1, 1}. The coin space is 4-dimensional.

## Step 1: Eigenspace Decomposition

Since τ_a² = I, define the projectors onto the ±1 eigenspaces:

    P_a^± = (I ± τ_a) / 2

These satisfy:
- P_a^+ + P_a^- = I
- P_a^+ P_a^- = 0
- (P_a^±)² = P_a^±
- (P_a^±)† = P_a^±
- τ_a = P_a^+ - P_a^-

So each term in the Hamiltonian decomposes as:

    τ_a D_a = (P_a^+ - P_a^-) D_a

## Step 2: Finite Differences

Following the 1D approach, use one-sided differences matched to eigenvalue sign.
Let δ = 2/3 (the face-step displacement magnitude). At site x with local
directions {e_a(x)}:

    P_a^+ D_a ψ(x) ≈ P_a^+ [ψ(x + δ e_a) - ψ(x)] / δ
    P_a^- D_a ψ(x) ≈ P_a^- [ψ(x) - ψ(x - δ e_a)] / δ

Combining:

    τ_a D_a ψ(x) ≈ [P_a^+ ψ(x + δ e_a) + P_a^- ψ(x - δ e_a) - ψ(x)] / δ

## Step 3: Shift Operators for Each Direction

Define a conditional shift for direction a:

    S_a ψ(x) = P_a^+(x) ψ(x + δ e_a(x)) + P_a^-(x) ψ(x - δ e_a(x))

where P_a^±(x) and e_a(x) are the projectors and directions at site x.
Then:

    τ_a D_a ψ ≈ (S_a - I) ψ / δ

and the full kinetic term:

    Σ_a τ_a D_a ψ ≈ Σ_a (S_a - I) ψ / δ

## Step 4: Time Evolution Operator

One time step of the Dirac equation:

    ψ(t+δ) ≈ (I - iδH) ψ = (I + Σ_a (S_a - I) - iδmβ) ψ

For an exactly unitary operator, we want a product of unitary factors.
Following the split-step approach:

    U = S_4 · S_3 · S_2 · S_1 · C

where C = exp(-iδmβ) is the coin/mass operator.

## Step 5: Unitarity Check for Individual S_a

**In the 1D case**, S = P₊T₊ + P₋T₋ is unitary because P₊P₋ = 0 and
T₊T₋ = I, giving SS† = P₊ + P₋ = I.

**For a single direction a**, if the projectors P_a^±(x) were the same
at every site, the same argument would give:

    S_a S_a† = P_a^+ T_{+a} T_{-a} + P_a^- T_{-a} T_{+a} = P_a^+ + P_a^- = I  ✓

But the projectors P_a^±(x) are **position-dependent** (the τ operators change
at each site due to the reflection rule). Let us examine what goes wrong.

### Computing S_a† S_a on a single chain

Consider sites along a chain connected by direction a. Label them by integer
n. At site n, the τ operator for direction a is τ_a^(n), with projectors
P_a^(n),±.

S_a sends:
- P_a^(n),+ component of ψ(n) → site n+1
- P_a^(n),- component of ψ(n) → site n-1

The new value at site n receives:
- P_a^(n-1),+ component of ψ(n-1) (from n-1, shifted forward)
- P_a^(n+1),- component of ψ(n+1) (from n+1, shifted backward)

So:  (S_a ψ)(n) = P_a^(n-1),+ ψ(n-1) + P_a^(n+1),- ψ(n+1)

Computing S_a† S_a:

    (S_a† S_a ψ)(n) = P_a^(n),+ (S_a ψ)(n+1) + P_a^(n),- (S_a ψ)(n-1)
                     = P_a^(n),+ [P_a^(n),+ ψ(n) + P_a^(n+2),- ψ(n+2)]
                     + P_a^(n),- [P_a^(n-2),+ ψ(n-2) + P_a^(n),- ψ(n)]

Wait — this is getting tangled because S_a† involves the SAME projectors.
Let me be more careful using the abstract notation.

Define S_a in Dirac notation on the full Hilbert space (coin ⊗ position):

    S_a = Σ_n  P_a^(n),+ ⊗ |n+1⟩⟨n|  +  P_a^(n),- ⊗ |n-1⟩⟨n|

Taking the adjoint (P^± are Hermitian, |n+1⟩⟨n|† = |n⟩⟨n+1|):

    S_a† = Σ_n  P_a^(n),+ ⊗ |n⟩⟨n+1|  +  P_a^(n),- ⊗ |n⟩⟨n-1|

Computing S_a† S_a:

    S_a† S_a = Σ_{n,m} [P_a^(n),+ P_a^(m),+ ⟨n+1|m+1⟩ ⊗ |n⟩⟨m|
                       + P_a^(n),+ P_a^(m),- ⟨n+1|m-1⟩ ⊗ |n⟩⟨m|
                       + P_a^(n),- P_a^(m),+ ⟨n-1|m+1⟩ ⊗ |n⟩⟨m|
                       + P_a^(n),- P_a^(m),- ⟨n-1|m-1⟩ ⊗ |n⟩⟨m|]

Evaluating the position inner products:

    = Σ_n [(P_a^(n),+)² ⊗ |n⟩⟨n|                           (m = n, terms 1&4)
          + (P_a^(n),-)² ⊗ |n⟩⟨n|
          + P_a^(n),+ P_a^(n+2),- ⊗ |n⟩⟨n+2|              (m = n+2, term 2)
          + P_a^(n),- P_a^(n-2),+ ⊗ |n⟩⟨n-2|]             (m = n-2, term 3)

Since (P^±)² = P^±:

    = Σ_n [(P_a^(n),+ + P_a^(n),-) ⊗ |n⟩⟨n|
          + P_a^(n),+ P_a^(n+2),- ⊗ |n⟩⟨n+2|
          + P_a^(n),- P_a^(n-2),+ ⊗ |n⟩⟨n-2|]

    = Σ_n [I_coin ⊗ |n⟩⟨n|
          + P_a^(n),+ P_a^(n+2),- ⊗ |n⟩⟨n+2|
          + P_a^(n),- P_a^(n-2),+ ⊗ |n⟩⟨n-2|]

### Unitarity condition

S_a† S_a = I  requires the off-diagonal terms to vanish:

    **P_a^(n),+ · P_a^(n+2),- = 0   for all n**

Expanding:  (I + τ_a^(n))(I - τ_a^(n+2)) = 0,  i.e.,  **τ_a^(n) = τ_a^(n+2)**

This says: the τ operator for direction a must be the same at sites n and n+2
(two steps apart along the chain).

**In the 1D case:** τ_a is the same at every site, so this is automatically
satisfied.

**In the 3D tetrahedral case:** The τ operators change at every step due to
the reflection rule. So this condition FAILS in general, and the naive S_a
is NOT unitary.

## Step 6: The BC Helix Approach

The BC helix decomposition offers a way to organize the shifts. Instead of
4 independent shift operators S_1,...,S_4 (one per direction), we define
2 shift operators S_R and S_L (one per spiral chirality).

Along an R-spiral with pattern [1, 3, 0, 2], each step uses a different
face index. At site n on the spiral, the face index is a(n) = pattern[n%4].

### The R-spiral shift operator

    S_R = Σ_n  P_{a(n)}^(n),+ ⊗ |n+1⟩⟨n|  +  P_{a(n)}^(n),- ⊗ |n-1⟩⟨n|

This is a 1D shift, but with a DIFFERENT τ operator at each site
(cycling through τ_1, τ_3, τ_0, τ_2 along the spiral).

### Unitarity of S_R

The same computation gives:

    S_R† S_R = I  +  Σ_n [P_{a(n)}^(n),+ · P_{a(n+2)}^(n+2),- ⊗ |n⟩⟨n+2|  +  h.c.]

The unitarity condition is now: P_{a(n)}^(n),+ · P_{a(n+2)}^(n+2),- = 0, i.e.,

    **τ_{a(n)}^(n) = τ_{a(n+2)}^(n+2)**

where a(n) = pattern[n%4] and a(n+2) = pattern[(n+2)%4].

For pattern [1,3,0,2]:
- n ≡ 0: need τ_1^(n) = τ_0^(n+2)
- n ≡ 1: need τ_3^(n) = τ_2^(n+2)
- n ≡ 2: need τ_0^(n) = τ_1^(n+2)
- n ≡ 3: need τ_2^(n) = τ_3^(n+2)

These are 4 conditions relating τ operators at sites 2 steps apart along
the spiral, for DIFFERENT face indices. We can check these numerically.

## Step 7: Numerical Verification

The unitarity condition τ_{a(n)}^(n) = τ_{a(n+2)}^(n+2) was checked
numerically along the R-spiral for the first 16 sites. Results:

- **No two τ operators on the helix are ever equal.** This is expected:
  the BC helix rotation angle arccos(-2/3) is irrational, so no two sites
  ever share the same orientation.

- The off-diagonal products P^(n,+) P^(n+2,-) have **rank 2** (out of 4),
  so the unitarity violation is not a small perturbation.

- The difference τ_{a(n)}^(n) - τ_{a(n+2)}^(n+2) always has **zero β
  component** (the β coefficient √7/4 is the same for all τ operators) but
  nonzero α components (the spatial parts differ because the local
  directions differ).

- This failure occurs for **all separations** k = 1, 2, ..., 8, not just
  k = 2. No pair of τ operators at any two sites on the helix are equal.

## Summary of Where We Stand

| Aspect                    | 1D case              | 3D tetrahedral case              |
|---------------------------|----------------------|----------------------------------|
| Coin space                | 2D (Pauli matrices)  | 4D (Dirac matrices)              |
| Shift operator            | S = P₊T₊ + P₋T₋     | S_R, S_L along BC helices        |
| Projectors                | Same at every site    | Position-dependent (τ_a changes) |
| Unitarity of shift        | Automatic (P₊P₋=0)   | Fails: τ_{a(n)}^(n) ≠ τ_{a(n+2)}^(n+2) |
| Walk operator             | U = S · C            | U = S_R · S_L · C (proposed)     |
| Continuum limit           | Dirac equation ✓     | Dirac equation (if unitary)      |

The central obstacle is the **position-dependence of the τ operators**,
which breaks the orthogonality argument that makes S unitary in the 1D case.
The irrational rotation angle of the BC helix guarantees that no two sites
share the same orientation, so the naive shift is never unitary.

### Possible resolutions

1. **Frame transport:** Include a unitary U_{n→n+1} in the shift that
   "rotates" the coin state from site n's frame to site n+1's frame.
   If U maps the eigenspaces of τ^(n) to those of τ^(n+1), the
   orthogonality structure could be restored.

2. **Different Hamiltonian splitting:** Instead of splitting H into
   individual τ_a D_a terms, find a decomposition aligned with the spiral
   structure where the shift operators are naturally unitary.

3. **Modified projectors:** Use projectors that are constant along the
   spiral (not tied to the local τ operators), while still reproducing
   the correct continuum limit.
