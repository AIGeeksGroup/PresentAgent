#!/bin/bash

# Define the root directory
root_dir="../../result/claude-3.7-sonnet"

# Use find to search for all .pptx files in the directory and subdirectories
find "$root_dir" -type f -name "*.pptx" | while IFS= read -r pptx; do
    echo "Running python test.py on \"$pptx\""
    python test.py --pptx "$pptx"
done
