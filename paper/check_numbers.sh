#!/bin/bash
# Pre-submission gate for unfilled results.
#
# Results for this paper are still being collected. Every number in main.tex is
# one of three states (see the "result-status machinery" block in the preamble):
#
#   \seedconf{v}     measured and seed-confirmed   -> silent
#   \oneseed{v}{tag} measured on one seed only     -> PROVISIONAL warning
#   \pending{tag}    not measured yet              -> PENDING warning
#
# Each non-final state writes a warning into main.log, so this script reads the
# build log rather than guessing from the source. Run it after latexmk.
#
# Usage:  ./check_numbers.sh            report status, exit 0
#         ./check_numbers.sh --strict   exit 1 if anything is unfilled
#
set -u
LOG="${LOG:-main.log}"
STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

if [ ! -f "$LOG" ]; then
  echo "no $LOG; run: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex"
  exit 2
fi

pend=$(grep -c "PENDING value:" "$LOG" || true)
prov=$(grep -c "PROVISIONAL single-seed" "$LOG" || true)
undef=$(grep -c "undefined on input" "$LOG" || true)
over=$(grep -c "Overfull" "$LOG" || true)
pages=$(grep -oE "Output written on .* \(([0-9]+) pages" "$LOG" | grep -oE "[0-9]+ pages" | tail -1)

echo "=== paper status ==="
printf "  %-28s %s\n" "pages"                "${pages:-unknown}"
printf "  %-28s %s\n" "PENDING (no value)"   "$pend"
printf "  %-28s %s\n" "PROVISIONAL (n=1)"    "$prov"
printf "  %-28s %s\n" "undefined refs/cites" "$undef"
printf "  %-28s %s\n" "overfull boxes"       "$over"

if [ "$pend" -gt 0 ]; then
  echo
  echo "--- values still unmeasured ---"
  grep -o "PENDING value: .*" "$LOG" | sed 's/PENDING value: /  /' | sed 's/ on input line.*//' | sort -u
fi

if [ "$prov" -gt 0 ]; then
  echo
  echo "--- single-seed values awaiting the seed mean ---"
  grep -o "PROVISIONAL single-seed value: .*" "$LOG" | sed 's/PROVISIONAL single-seed value: /  /' | sed 's/ on input line.*//' | sort -u
fi

echo
if [ "$pend" -eq 0 ] && [ "$prov" -eq 0 ] && [ "$undef" -eq 0 ] && [ "$over" -eq 0 ]; then
  echo "READY: every number is measured and seed-confirmed."
  echo "Set \\draftnumbersfalse in the preamble for the submission build."
  exit 0
fi

echo "NOT READY for submission."
[ "$STRICT" -eq 1 ] && exit 1
exit 0
