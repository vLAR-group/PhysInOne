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
    printf '\r%-100s\n' ' '
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

print_progress() {
    local completed=$1
    local uid=$2
    local percent filled empty bar spaces

    percent=$((completed * 100 / TOTAL))
    filled=$((completed * BAR_WIDTH / TOTAL))
    empty=$((BAR_WIDTH - filled))
    bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
    spaces=$(printf '%*s' "$empty" '')

    printf '\rProgress: [%s%s] %3d%% (%d/%d) UID: %-16s [Q = quit]' \
        "$bar" "$spaces" "$percent" "$completed" "$TOTAL" "$uid"
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
if [[ -t 0 ]]; then
    echo "Control: press Q or q at any time to quit; the active partial ZIP will be removed."
else
    echo "Control: Q-to-quit is unavailable because standard input is not an interactive terminal."
fi

BAR_WIDTH=40
SUCCEEDED=0
FAILED=0
CANCELLED=0

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

printf '\r%-100s\n' ' '

if (( CANCELLED )); then
    echo "Packaging cancelled by user."
    echo "Completed archives: $SUCCEEDED"
    echo "Output directory: $OUTPUT_DIR"
    exit 130
fi

echo "Packaging finished: $SUCCEEDED succeeded, $FAILED failed."
echo "Output directory: $OUTPUT_DIR"

(( FAILED == 0 ))
