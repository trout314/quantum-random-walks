# 1D Dirac Equation as a Quantum Walk

## The 1D Dirac Equation

The 1D Dirac equation for a 2-component spinor ψ = (ψ₊, ψ₋)ᵀ:

    i ∂ₜψ = H ψ,    H = σ_z p + m σ_x

where p = -i∂ₓ, so explicitly:

    i ∂ₜψ = (-i σ_z ∂ₓ + m σ_x) ψ

Written out in components with σ_z = diag(1,-1):

    i ∂ₜψ₊ = -i ∂ₓψ₊ + m ψ₋
    i ∂ₜψ₋ =  i ∂ₓψ₋ + m ψ₊

So ψ₊ is a right-mover and ψ₋ is a left-mover, coupled by the mass term.

## Discretization

Put ψ on a 1D lattice with spacing ε. Use time step ε (setting c = 1).

### Step 1: Split H into kinetic and mass terms

    H = H_kin + H_mass,    H_kin = σ_z p = -i σ_z ∂ₓ,    H_mass = m σ_x

### Step 2: Approximate the kinetic term with finite differences

The key idea: σ_z = P₊ - P₋ where P± = (I ± σ_z)/2 are the projectors
onto the spin-up (right-mover) and spin-down (left-mover) components.
So:

    σ_z ∂ₓψ = P₊ ∂ₓψ - P₋ ∂ₓψ

Use a **one-sided finite difference matched to the propagation direction**:

    P₊ ∂ₓψ(x) ≈ P₊ [ψ(x+ε) - ψ(x)] / ε      (forward difference for right-movers)
    P₋ ∂ₓψ(x) ≈ P₋ [ψ(x) - ψ(x-ε)] / ε      (backward difference for left-movers)

Therefore:

    σ_z ∂ₓψ(x) ≈ [P₊ ψ(x+ε) + P₋ ψ(x-ε) - ψ(x)] / ε

### Step 3: Write in terms of shift operators

Define:

    T₊ ψ(x) = ψ(x + ε)      (shift right)
    T₋ ψ(x) = ψ(x - ε)      (shift left)

and the **conditional shift operator**:

    S = P₊ ⊗ T₊ + P₋ ⊗ T₋

This shifts the right-mover component one step right and the left-mover
component one step left. Then:

    S ψ(x) = P₊ ψ(x+ε) + P₋ ψ(x-ε)

and the kinetic finite difference becomes:

    σ_z ∂ₓψ ≈ (S - I) ψ / ε

### Step 4: Construct the time evolution operator

For one time step, the Dirac equation gives:

    ψ(t+ε) ≈ (I - iε H) ψ(t) = (I + ε σ_z ∂ₓ - iεm σ_x) ψ(t)

Substituting the finite difference:

    ψ(t+ε) ≈ (I + (S - I) - iεm σ_x) ψ(t) = (S - iεm σ_x) ψ(t)

But S - iεm σ_x is only approximately unitary (to first order in ε). We want
an **exactly unitary** evolution operator. The standard trick: implement the
mass term as a unitary coin operator.

Define the **coin operator**:

    C = e^{-iεm σ_x} = cos(εm) I - i sin(εm) σ_x

This is manifestly unitary (exponential of i times Hermitian matrix).

The walk operator is:

    U = S · C

One time step: ψ(t+ε) = U ψ(t) = S · C · ψ(t).

## Unitarity of U

**Claim:** U = S · C is unitary.

**Proof:** It suffices to show S and C are each unitary.

*C is unitary:* σ_x is Hermitian, so C = e^{-iεmσ_x} satisfies
C† = e^{+iεmσ_x}, and C C† = C† C = I. ✓

*S is unitary:* We compute S† and verify S S† = S† S = I.

    S† = (P₊ ⊗ T₊ + P₋ ⊗ T₋)† = P₊† ⊗ T₊† + P₋† ⊗ T₋†
       = P₊ ⊗ T₋ + P₋ ⊗ T₊

(using P±† = P± since projectors are Hermitian, and T±† = T∓ since
shifting right and left are adjoint operations.)

Now:

    S S† = (P₊ T₊ + P₋ T₋)(P₊ T₋ + P₋ T₊)
         = P₊P₊ T₊T₋ + P₊P₋ T₊T₊ + P₋P₊ T₋T₋ + P₋P₋ T₋T₊

Using the projector identities P₊P₊ = P₊, P₋P₋ = P₋, P₊P₋ = P₋P₊ = 0,
and the shift identities T₊T₋ = T₋T₊ = I:

    S S† = P₊ · I + 0 + 0 + P₋ · I = P₊ + P₋ = I   ✓

Similarly S† S = I. So S is unitary.

Therefore U = S · C is unitary (product of unitaries). ✓

## Continuum limit

Taylor-expanding U = S · C in powers of ε:

    S = P₊ T₊ + P₋ T₋
      = P₊(I + ε∂ₓ + ½ε²∂ₓ² + ...) + P₋(I - ε∂ₓ + ½ε²∂ₓ² + ...)
      = I + ε σ_z ∂ₓ + ½ε² ∂ₓ² + O(ε³)

    C = I - iεm σ_x + O(ε²)

    U = S · C = (I + ε σ_z ∂ₓ + ...)(I - iεm σ_x + ...)
      = I + ε σ_z ∂ₓ - iεm σ_x + O(ε²)
      = I + iε(-i σ_z ∂ₓ + m σ_x) + O(ε²)...

Wait — let's be careful with signs. We have H = -iσ_z ∂ₓ + m σ_x, so:

    U ≈ I + ε σ_z ∂ₓ - iεm σ_x = I + iε(σ_z p) - iε(m σ_x) = I + iε(σ_z p - m σ_x)

This means ψ(t+ε) = U ψ(t) ≈ (I + iεH')ψ(t) where H' = σ_z p - m σ_x.

Taking the continuum limit:

    i ∂ₜψ = -H' ψ = (-σ_z p + m σ_x) ψ

This is the 1D Dirac equation with the sign convention H = -σ_z p + m σ_x,
which is physically equivalent (just a spatial reflection x → -x, or
equivalently relabeling left↔right movers). The dispersion relation is
the same: E² = p² + m².

## Summary

| Dirac equation piece       | Walk operator piece          |
|-----------------------------|------------------------------|
| Kinetic term σ_z p          | Conditional shift S = P₊T₊ + P₋T₋ |
| Mass term m σ_x             | Coin operator C = e^{-iεmσ_x}  |
| Time evolution e^{-iεH}     | Walk step U = S · C          |
| Unitarity of e^{-iεH}       | S†S = I (projector orthogonality + shift inverses), C†C = I |
