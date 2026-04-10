#!/bin/bash
# Test walk memory usage at various sigma values.
# Lattice grows on demand, limited only by the 4GB RAM floor.
# Uses /usr/bin/time to measure peak RSS.
# Args: nsteps mix_phi

WALK_BIN="$(dirname "$0")/../../dlang/build/walk_d"
NSTEPS=${1:-20}
PHI=${2:-0.03}
THRESH="1e-10"
PRUNE="1e-6"
SEED_THRESH="1e-4"
DMIN="0.35"

echo "=== Memory scaling test: nsteps=$NSTEPS phi=$PHI dMin=$DMIN ==="
echo ""
printf "%-8s  %-10s  %-12s  %-10s  %s\n" "sigma" "peakRSS_MB" "final_sites" "norm" "status"
echo "----------------------------------------------------------------------"

for SIGMA in 3 5 10 15 20; do
    # Args: theta sigma nsteps threshold pruneThresh mixPhi seedThresh dMin
    OUTPUT=$( /usr/bin/time -v "$WALK_BIN" 0 "$SIGMA" "$NSTEPS" "$THRESH" "$PRUNE" "$PHI" "$SEED_THRESH" "$DMIN" 2>&1 )
    EXIT_CODE=$?

    # Extract peak RSS from time output (in KB)
    PEAK_KB=$(echo "$OUTPUT" | grep "Maximum resident" | awk '{print $NF}')
    PEAK_MB=$(( PEAK_KB / 1024 ))

    # Extract last data line for final norm and site count
    LAST_LINE=$(echo "$OUTPUT" | grep -E '^[0-9]+ ' | tail -1)
    if [ -n "$LAST_LINE" ]; then
        NORM=$(echo "$LAST_LINE" | awk '{print $2}')
        NSITES=$(echo "$LAST_LINE" | awk '{print $11}')
    else
        NORM="?"
        NSITES="?"
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="OK"
    elif [ $EXIT_CODE -eq 137 ]; then
        STATUS="OOM-KILLED"
    else
        STATUS="EXIT=$EXIT_CODE"
    fi

    printf "%-8s  %-10s  %-12s  %-10s  %s\n" "$SIGMA" "${PEAK_MB}MB" "$NSITES" "$NORM" "$STATUS"

    # Stop if we got killed
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "Stopped at sigma=$SIGMA (exit=$EXIT_CODE)"
        break
    fi
done
