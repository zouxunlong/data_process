#!/bin/bash

# Top-level directory to search
top_dir="./"

# Find all subdirectories
find "$top_dir" -type d -maxdepth 1 | while read -r dir
do
    # Count the number of files recursively in each directory
    count=$(find "$dir" -type f | wc -l)
    echo "${dir} has ${count} files. abcdahjkewadnasj"
done