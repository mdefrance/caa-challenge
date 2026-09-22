set -e
cd /c/Users/defra/Desktop/git/PROJECTS/caa-challenge
LOCK=.sharpen.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: another sharpen queue is running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
R="uv run --no-sync python tools/ablation_matrix.py"
step () { echo "=== $1 start $(date +%T) ==="; shift; "$@"; }
# frequency AutoCarver skipped: identical to the own-selection arm by construction,
# verified exactly on seeds 42 and 1. The severity AutoCarver arm still runs, because
# only the matrix harness computes the ranking metrics.
step "ORD-FREQ-OB" $R --model frequency --arm optbinning --features fixed --tag fixed_optbinning
step "ORD-FREQ-NC" $R --model frequency --arm nocarve    --features fixed --tag fixed_nocarve
step "ORD-AMT-AC"  $R --model amount    --arm autocarver --features fixed --tag fixed_autocarver
step "ORD-AMT-OB"  $R --model amount    --arm optbinning --features fixed --tag fixed_optbinning
step "ORD-AMT-NC"  $R --model amount    --arm nocarve    --features fixed --tag fixed_nocarve
echo "=== ALL DONE $(date +%T) ==="
