#!/bin/bash

# Script to remove all files/folders inside specified directories
# while preserving the directories themselves

# Define directories to clean
DIRS=("tmp" "sources/workflows" "sources/memory")

# Function to clean a directory
clean_directory() {
    local dir="$1"
    
    # Check if directory exists
    if [ -d "$dir" ]; then
        echo "Cleaning directory: $dir"
        
        # Remove all contents while preserving the directory itself
        find "$dir" -mindepth 1 -delete
        
        echo "✓ Directory cleaned successfully: $dir"
    else
        echo "⚠️ Directory does not exist: $dir (skipping)"
    fi
}

# Main execution
echo "Starting cleanup process..."

for dir in "${DIRS[@]}"; do
    clean_directory "$dir"
done

echo "Cleanup process completed."