#!/usr/bin/env bash
set -euo pipefail

# Ensure ffmpeg is installed
command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg is not installed. Please install it via 'brew install ffmpeg' and retry." >&2; exit 1; }

# Enable nullglob so the glob turns into an empty array if no files match
shopt -s nullglob
files=( *.mov )

# If no .mov files, exit
if [ ${#files[@]} -eq 0 ]; then
  echo "No .mov files found in the current directory."
  exit 0
fi

# Counter for sequential naming
count=1
for f in "${files[@]}"; do
  echo "Converting '$f' to '${count}.mp4'..."
  # Remux into MP4 container (fast). Change to -c:v libx264 -c:a aac if re-encode needed
  ffmpeg -i "$f" -c copy "${count}.mp4"
  count=$((count + 1))
done

# Delete original .mov files
echo "Conversion complete. Deleting original .mov files..."
rm -f -- *.mov

echo "All done! Converted ${count}-1 files and cleaned up .movs."

