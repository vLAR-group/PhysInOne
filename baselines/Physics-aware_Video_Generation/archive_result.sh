#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_search_dir> <output_submission_dir>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Find and count total items first to calculate progress
echo "Scanning for matching directories..."
MAPFILE=()
while IFS= read -r -d '' line; do
    MAPFILE+=("$line")
done < <(find "$INPUT_DIR" -type d -path "*__bg*__*_trajectory" -print0)

TOTAL=${#MAPFILE[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "❌ No folders matching *__bg*__*_trajectory found under $INPUT_DIR"
    exit 0
fi

echo "📦 Found $TOTAL directories to package. Starting..."

# Progress bar layout configuration
BAR_WIDTH=40

for ((i=0; i<TOTAL; i++)); do
    TARGET_DIR="${MAPFILE[i]}"
    FOLDER_NAME=$(basename "$TARGET_DIR")
    ZIP_PATH="$OUTPUT_DIR/${FOLDER_NAME}.zip"
    
    # Package the directory silently
    PARENT_DIR=$(dirname "$TARGET_DIR")
    (cd "$PARENT_DIR" && zip -rq "$ZIP_PATH" "$FOLDER_NAME")
    
    # Calculate progress bar percentages
    CURRENT=$((i + 1))
    PERCENT=$(( CURRENT * 100 / TOTAL ))
    FILLED=$(( CURRENT * BAR_WIDTH / TOTAL ))
    EMPTY=$(( BAR_WIDTH - FILLED ))
    
    # Build the bar visuals
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')
    SPACES=$(printf "%${EMPTY}s")
    
    # Output the updating line (\r keeps it rewriting the same line)
    printf "\rProgress: [%s%s] %d%% (%d/%d) Processing: %.30s..." "$BAR" "$SPACES" "$PERCENT" "$CURRENT" "$TOTAL" "$FOLDER_NAME"
done

# Clear line and print final status
printf "\r%-100s\n" " "
echo "✅ Packaging complete! All zip files are located in: $OUTPUT_DIR"
