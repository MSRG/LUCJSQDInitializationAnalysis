#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

while IFS= read -r i; do
    #sbatch --chdir="${SCRIPT_DIR}/postprocess" "$i"
    echo "$i"
done < "${SCRIPT_DIR}/rerun.sh"
