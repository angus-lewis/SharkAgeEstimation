#!/usr/bin/env bash

src="./perf/gp/"
dst="./perf/gp1/"

# oldpart="length1024"
# newpart="length2048"

find "$src" -type f -name "config.*length256*.yaml" | while read -r file; do
  # Get relative path (remove source prefix)
  rel_path="${file#$src/}"

  # Extract directory and filename
  dir=$(dirname "$rel_path")
  filename=$(basename "$rel_path")

  # Replace part of filename
#   newname="${filename//oldpart/newpart}"

  # Create destination directory
  mkdir -p "$dst/$dir"/
  echo "$dst/$dir"

  # Copy file with new name
#   out="$dst/$dir/$newname"
  out="$dst/$dir/$filename"
  cp "$file" "$dst/$dir/$newname"
  echo "$out"

  # in-place edit after copy
#   sed -i '' 's/n_sims: 0/n_sims: 400/g' "$out"

    # base_seed="../../seeds/signal_gen_seeds.txt"

    # awk -v seed_path="$base_seed" '
    # BEGIN { in_sg=0; skip=0 }

    # /^signal_generator:/ {
    #     in_sg=1
    #     print
    #     next
    # }

    # # Replace the method block inside signal_generator
    # in_sg && /^[[:space:]]+method:/ {
    #     print "  method:"
    #     print "    name: piecewise_linear"
    #     print "    n_peaks: 8"
    #     print "    min_peak_size: 1.0"
    #     print "    peak_size_shape: 1"
    #     print "    peak_size_rate: 1"
    #     print "    rng_seed: " seed_path
    #     print "    min_distance_between_peaks: 3.5"
    #     skip=1
    #     next
    # }

    # # Stop skipping when we leave the method block
    # skip && /^[[:space:]]{2}[^[:space:]]/ {
    #     skip=0
    # }

    # skip { next }

    # { print }
    # ' "$out" > "$out.tmp" && mv "$out.tmp" "$out"
done

# cd "${dst}"
# find . -type f \( -name "*.yml" -o -name "*.yaml" \) -exec sed -i '' 's|/seeds/|/seeds_med/|g' {} +