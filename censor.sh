#!/usr/bin/env bash
set -euo pipefail

# Script: censor_all.sh
# Description: Uses the Python censor script (censor_billboards.py) to process every .mp4 in the current directory.
# Outputs censored1.mp4, censored2.mp4, etc.

# Ensure the Python script exists
PYTHON_SCRIPT="video.py"
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is not installed." >&2; exit 1; }
[ -f "$PYTHON_SCRIPT" ] || { echo "Error: $PYTHON_SCRIPT not found in current directory." >&2; exit 1; }

# Enable nullglob so the glob turns into an empty array if no files match
shopt -s nullglob
mp4s=( *.mp4 )

if [ ${#mp4s[@]} -eq 0 ]; then
  echo "No .mp4 files found to censor."
  exit 0
fi

count=1
for infile in "${mp4s[@]}"; do
  outfile="censored${count}.mp4"
  echo "Censoring '$infile' -> '$outfile'..."
  python3 "$PYTHON_SCRIPT" "$infile" "$outfile"
  count=$((count + 1))
done

echo "All done! Censored ${count}-1 files."
