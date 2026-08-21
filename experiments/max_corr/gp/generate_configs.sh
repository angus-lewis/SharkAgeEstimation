#!/bin/bash

find . -type f -name "_*.yaml" -exec bash -c '
for f; do
  dir=$(dirname "$f")
  base=$(basename "$f")
  mv -- "$f" "$dir/${base#_}"
done
' bash {} +