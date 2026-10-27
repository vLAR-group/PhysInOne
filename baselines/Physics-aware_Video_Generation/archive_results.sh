#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_search_dir> <output_submission_dir>" >&2
    exit 1
fi

if [[ ! -d $1 ]]; then
    echo "Error: input directory does not exist: $1" >&2
    exit 1
fi

# Resolve both paths before changing directories during packaging.
INPUT_DIR=$(cd -- "$1" && pwd -P)
mkdir -p -- "$2" || {
    echo "Error: could not create output directory: $2" >&2
    exit 1
}
OUTPUT_DIR=$(cd -- "$2" && pwd -P)

ACTIVE_ZIP_PID=""
ACTIVE_ZIP_PATH=""
ACTIVE_ZIP_LOG=""

cleanup_on_signal() {
    if [[ -n $ACTIVE_ZIP_PID ]] && kill -0 "$ACTIVE_ZIP_PID" 2>/dev/null; then
        kill "$ACTIVE_ZIP_PID" 2>/dev/null || true
        wait "$ACTIVE_ZIP_PID" 2>/dev/null || true
    fi
    if [[ -n $ACTIVE_ZIP_PATH ]]; then
        rm -f -- "$ACTIVE_ZIP_PATH"
    fi
    if [[ -n $ACTIVE_ZIP_LOG ]]; then
        rm -f -- "$ACTIVE_ZIP_LOG"
    fi
    printf '\r%-160s\n' ' '
    echo "Packaging interrupted. The incomplete ZIP was removed."
    exit 130
}

trap cleanup_on_signal INT TERM

scene_uid() {
    # Example:
    #   ...__bg180__1Fievf_trajectory -> 1Fievf
    local name=$1
    name=${name%_trajectory}
    if [[ $name == *"__"* ]]; then
        printf '%s' "${name##*__}"
    else
        printf '%s' "$name"
    fi
}

format_duration() {
    local total_seconds=$1
    local hours minutes seconds

    (( total_seconds < 0 )) && total_seconds=0
    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    seconds=$((total_seconds % 60))

    if (( hours > 0 )); then
        printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
    else
        printf '%02d:%02d' "$minutes" "$seconds"
    fi
}

print_progress() {
    local completed=$1
    local uid=$2
    local percent filled empty bar spaces now elapsed eta elapsed_text eta_text

    percent=$((completed * 100 / TOTAL))
    filled=$((completed * BAR_WIDTH / TOTAL))
    empty=$((BAR_WIDTH - filled))
    bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
    spaces=$(printf '%*s' "$empty" '')

    now=$(date +%s)
    elapsed=$((now - START_TIME))
    elapsed_text=$(format_duration "$elapsed")
    if (( completed > 0 )); then
        eta=$((elapsed * (TOTAL - completed) / completed))
        eta_text=$(format_duration "$eta")
    else
        eta_text="calculating..."
    fi

    printf '\rProgress: [%s%s] %3d%% (%d/%d) Elapsed: %s | ETA: %-14s UID: %-16s [Q = quit]' \
        "$bar" "$spaces" "$percent" "$completed" "$TOTAL" \
        "$elapsed_text" "$eta_text" "$uid"
}

echo "Scanning recursively for matching trajectory directories..."
mapfile -d '' -t TARGETS < <(
    find "$INPUT_DIR" -type d -name '*__bg*__*_trajectory' -print0
)

TOTAL=${#TARGETS[@]}
if (( TOTAL == 0 )); then
    echo "No folders matching *__bg*__*_trajectory found under $INPUT_DIR"
    exit 0
fi

echo "Found $TOTAL directories to package."
echo "Output layout: all scene ZIP files will be written directly under the output root."

# Flattening removes source parent folders. Refuse to continue if two source
# scenes have the same folder name, because they would map to the same ZIP.
declare -A SEEN_SCENE_NAMES=()
DUPLICATE_NAMES=()
for TARGET_DIR in "${TARGETS[@]}"; do
    FOLDER_NAME=$(basename -- "$TARGET_DIR")
    if [[ -n ${SEEN_SCENE_NAMES[$FOLDER_NAME]+present} ]]; then
        DUPLICATE_NAMES+=("$FOLDER_NAME")
    else
        SEEN_SCENE_NAMES[$FOLDER_NAME]=$TARGET_DIR
    fi
done

if (( ${#DUPLICATE_NAMES[@]} > 0 )); then
    echo "Error: cannot flatten output because duplicate scene folder names exist:" >&2
    printf '  %s\n' "${DUPLICATE_NAMES[@]}" >&2
    echo "Rename or remove duplicates, then run the script again." >&2
    exit 1
fi

if [[ -t 0 ]]; then
    echo "Control: press Q or q at any time to quit; the active partial ZIP will be removed."
else
    echo "Control: Q-to-quit is unavailable because standard input is not an interactive terminal."
fi

BAR_WIDTH=40
SUCCEEDED=0
FAILED=0
CANCELLED=0
START_TIME=$(date +%s)

for ((i = 0; i < TOTAL; i++)); do
    TARGET_DIR=${TARGETS[i]}
    FOLDER_NAME=$(basename -- "$TARGET_DIR")
    UID_CODE=$(scene_uid "$FOLDER_NAME")
    ZIP_PATH="$OUTPUT_DIR/${FOLDER_NAME}.zip"
    ZIP_ERROR_LOG="$OUTPUT_DIR/.${FOLDER_NAME}.zip-error.log"

    print_progress "$i" "$UID_CODE"

    # Create a clean archive containing the scene directory's contents only.
    # This places CineCamera_*/ directly at the ZIP root and avoids a
    # same-named outer scene folder.
    rm -f -- "$ZIP_PATH" "$ZIP_ERROR_LOG"
    ACTIVE_ZIP_PATH=$ZIP_PATH
    ACTIVE_ZIP_LOG=$ZIP_ERROR_LOG
    (
        cd -- "$TARGET_DIR" || exit 1
        exec zip -rq "$ZIP_PATH" .
    ) >"$ZIP_ERROR_LOG" 2>&1 &
    ACTIVE_ZIP_PID=$!

    if [[ -t 0 ]]; then
        while kill -0 "$ACTIVE_ZIP_PID" 2>/dev/null; do
            if IFS= read -r -s -n 1 -t 0.1 key; then
                case $key in
                    q|Q)
                        CANCELLED=1
                        kill "$ACTIVE_ZIP_PID" 2>/dev/null || true
                        wait "$ACTIVE_ZIP_PID" 2>/dev/null || true
                        rm -f -- "$ZIP_PATH" "$ZIP_ERROR_LOG"
                        break
                        ;;
                esac
            fi
        done
    fi

    if (( CANCELLED )); then
        ACTIVE_ZIP_PID=""
        ACTIVE_ZIP_PATH=""
        ACTIVE_ZIP_LOG=""
        break
    fi

    if wait "$ACTIVE_ZIP_PID"; then
        ((SUCCEEDED += 1))
        rm -f -- "$ZIP_ERROR_LOG"
    else
        ((FAILED += 1))
        rm -f -- "$ZIP_PATH"
        printf '\nError: failed to create %s\n' "$ZIP_PATH" >&2
        if [[ -s $ZIP_ERROR_LOG ]]; then
            sed 's/^/  /' "$ZIP_ERROR_LOG" >&2
        fi
        rm -f -- "$ZIP_ERROR_LOG"
    fi

    ACTIVE_ZIP_PID=""
    ACTIVE_ZIP_PATH=""
    ACTIVE_ZIP_LOG=""
    print_progress "$((i + 1))" "$UID_CODE"
done

printf '\r%-160s\n' ' '
TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_ELAPSED_TEXT=$(format_duration "$TOTAL_ELAPSED")

if (( CANCELLED )); then
    echo "Packaging cancelled by user."
    echo "Completed archives: $SUCCEEDED"
    echo "Elapsed time: $TOTAL_ELAPSED_TEXT"
    echo "Output directory: $OUTPUT_DIR"
    exit 130
fi

echo "Packaging finished: $SUCCEEDED succeeded, $FAILED failed."
echo "Elapsed time: $TOTAL_ELAPSED_TEXT"
echo "Output directory: $OUTPUT_DIR"

(( FAILED == 0 ))
