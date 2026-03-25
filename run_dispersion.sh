#!/bin/bash
# Collect 3D dispersion relation data.
# Runs walk_adaptive at various sigma values, massless and massive.
# Expected runtime: ~12 hours total.
#
# Output: /tmp/dispersion_3d/ directory with one file per run.

set -e
OUTDIR=/tmp/dispersion_3d
mkdir -p "$OUTDIR"
WALK=./walk_adaptive
THRESH=1e-10
COIN=1  # dual parity

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTDIR/log.txt"; }

log "=== Dispersion relation collection started ==="
log "MAX_SITES=60M, chain-first seeding, max_per_cell=8"

# Massless runs (theta=0, mix_phi=0): measure speed of light c
# Larger sigma = cleaner continuum limit but more sites needed
# At ~350k new sites/step for sigma=5, 60M gives ~170 steps
# For sigma=3, seed is smaller so we get more steps
for sigma in 3.0 4.0 5.0 6.0 8.0; do
    # Estimate max steps: seed ~ (4*sigma/0.667)^3 * density ~= sigma^3 * 200
    # New sites/step ~ sigma^2 * 30 (wavefront area)
    # Available steps ~ (60M - seed) / (new_per_step)
    nsteps=200  # will stop early if site limit hit
    tag="massless_s${sigma}"
    outf="$OUTDIR/${tag}.dat"
    errf="$OUTDIR/${tag}.err"
    log "Starting $tag: sigma=$sigma steps=$nsteps"
    $WALK 0 $sigma $nsteps $THRESH 0 $COIN 0.0 > "$outf" 2> "$errf" || true
    last=$(grep "^[0-9]" "$outf" | tail -1)
    log "  Done $tag: $(echo "$last" | awk '{print "t="$1, "norm="$2, "r2="$3, "sites="$8}')"
done

# Massive runs with V mixing: measure v(C) at different C = phi * sigma
# Use sigma=5 as baseline
sigma=5.0
for phi in 0.01 0.02 0.05 0.1 0.2; do
    C=$(python3 -c "print(f'{$phi * $sigma:.2f}')")
    nsteps=150
    tag="massive_s${sigma}_phi${phi}_C${C}"
    outf="$OUTDIR/${tag}.dat"
    errf="$OUTDIR/${tag}.err"
    log "Starting $tag: sigma=$sigma phi=$phi C=$C steps=$nsteps"
    $WALK 0 $sigma $nsteps $THRESH 0 $COIN $phi > "$outf" 2> "$errf" || true
    last=$(grep "^[0-9]" "$outf" | tail -1)
    log "  Done $tag: $(echo "$last" | awk '{print "t="$1, "norm="$2, "r2="$3, "sites="$8}')"
done

# Repeat massive runs at sigma=3 (smaller seed, more steps possible)
sigma=3.0
for phi in 0.02 0.05 0.1 0.2 0.5; do
    C=$(python3 -c "print(f'{$phi * $sigma:.2f}')")
    nsteps=300
    tag="massive_s${sigma}_phi${phi}_C${C}"
    outf="$OUTDIR/${tag}.dat"
    errf="$OUTDIR/${tag}.err"
    log "Starting $tag: sigma=$sigma phi=$phi C=$C steps=$nsteps"
    $WALK 0 $sigma $nsteps $THRESH 0 $COIN $phi > "$outf" 2> "$errf" || true
    last=$(grep "^[0-9]" "$outf" | tail -1)
    log "  Done $tag: $(echo "$last" | awk '{print "t="$1, "norm="$2, "r2="$3, "sites="$8}')"
done

log "=== All runs complete ==="
log "Results in $OUTDIR/"
