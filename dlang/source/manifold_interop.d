/**
 * manifold_interop.d — C interface for building and running a manifold walk.
 *
 * Python creates the lattice structure (sites, closed chains) via these
 * functions, then runs the walk loop in D for performance.
 */
module manifold_interop;

import geometry : Vec3;
import dirac : Mat4, makeTau, projPlus, projMinus, frameTransport, mul, matVecSplit;
import lattice : Lattice;
import operators : pureShift, applyVmix;

/// Opaque handle for the lattice.  We store a heap-allocated pointer
/// so the GC keeps it alive across C calls.
private Lattice!false* gLat;

// ---- Lattice lifecycle ----

export extern(C)
void manifold_create(int capacity) {
    gLat = new Lattice!false;
    *gLat = Lattice!false.create(capacity);
}

export extern(C)
int manifold_nsites() {
    return gLat.nsites;
}

export extern(C)
int manifold_nchains() {
    return cast(int) gLat.chains.length;
}

// ---- Site creation ----

/// Allocate a site at (x, y, z). Returns site ID.
export extern(C)
int manifold_alloc_site(double x, double y, double z) {
    return gLat.allocSite(Vec3(x, y, z));
}

// ---- Chain creation ----

/// Create a closed chain from a list of site IDs and per-site exit directions.
///
/// siteIds:  array of nSites site IDs in chain order
/// exitDirs: array of 3*nSites doubles (x,y,z for each site's exit direction)
/// isR:      true for R-chain, false for L-chain
///
/// Builds all fwdBlock/bwdBlock ops and closes the chain (wrap-around).
/// Links each site to this chain.
/// Returns the chain ID.
export extern(C)
int manifold_add_closed_chain(const(int)* siteIds, const(double)* exitDirs,
                              int nSites, bool isR) {
    alias Ops = Lattice!false.Ops;

    // Create chain
    int chainId = cast(int) gLat.chains.length;
    Lattice!false.ChainT newChain;
    newChain.isR = isR;
    newChain.isClosed = true;
    newChain.rootSite = siteIds[0];
    newChain.rootIdx = 0;

    // Build ops for each site
    foreach (i; 0 .. nSites) {
        int sid = siteIds[i];
        Vec3 d = Vec3(exitDirs[3*i], exitDirs[3*i+1], exitDirs[3*i+2]);
        Mat4 tau = makeTau(d);

        int prevIdx = (i > 0) ? i - 1 : nSites - 1;
        int nextIdx = (i < nSites - 1) ? i + 1 : 0;
        Vec3 dPrev = Vec3(exitDirs[3*prevIdx], exitDirs[3*prevIdx+1], exitDirs[3*prevIdx+2]);
        Vec3 dNext = Vec3(exitDirs[3*nextIdx], exitDirs[3*nextIdx+1], exitDirs[3*nextIdx+2]);
        Mat4 tauPrev = makeTau(dPrev);
        Mat4 tauNext = makeTau(dNext);

        Ops op;
        op.siteId = sid;
        op.fwdBlock = mul(frameTransport(tau, tauNext), projPlus(tau));
        op.bwdBlock = mul(frameTransport(tau, tauPrev), projMinus(tau));
        newChain.ops.pushBack(op);

        // Link site to chain
        if (isR) { gLat.sites[sid].rChain = chainId; gLat.sites[sid].rIdx = i; }
        else     { gLat.sites[sid].lChain = chainId; gLat.sites[sid].lIdx = i; }
    }

    gLat.chains ~= newChain;
    return chainId;
}

// ---- Wavefunction access ----

/// Set spinor at site s: psi[s] = (re[0]+i*im[0], ..., re[3]+i*im[3])
export extern(C)
void manifold_set_psi(int siteId, const(double)* re, const(double)* im) {
    gLat.psiRe[4*siteId .. 4*siteId+4] = re[0..4];
    gLat.psiIm[4*siteId .. 4*siteId+4] = im[0..4];
}

/// Read all psi into caller-provided buffers (4*nsites doubles each).
export extern(C)
void manifold_get_psi(double* outRe, double* outIm) {
    int n4 = 4 * gLat.nsites;
    outRe[0 .. n4] = gLat.psiRe[0 .. n4];
    outIm[0 .. n4] = gLat.psiIm[0 .. n4];
}

/// Compute total norm² of the wavefunction.
export extern(C)
double manifold_norm2() {
    double n2 = 0;
    int n4 = 4 * gLat.nsites;
    foreach (i; 0 .. n4)
        n2 += gLat.psiRe[i]*gLat.psiRe[i] + gLat.psiIm[i]*gLat.psiIm[i];
    return n2;
}

// ---- Walk step ----

/// Run one full walk step: S_L → Vmix_L → S_R → Vmix_R.
/// Returns norm² after the step.
export extern(C)
double manifold_step(double mixPhi) {
    // S_L (shift along L-chains, no overflow for closed chains)
    pureShift!false(*gLat, false);  // isR=false → L-chains
    applyVmix!false(*gLat, false, mixPhi);

    // S_R (shift along R-chains)
    pureShift!false(*gLat, true);   // isR=true → R-chains
    applyVmix!false(*gLat, true, mixPhi);

    return manifold_norm2();
}

/// Run nSteps walk steps, writing norm² at each step to outNorms.
export extern(C)
void manifold_run(int nSteps, double mixPhi, double* outNorms) {
    foreach (t; 0 .. nSteps)
        outNorms[t] = manifold_step(mixPhi);
}
