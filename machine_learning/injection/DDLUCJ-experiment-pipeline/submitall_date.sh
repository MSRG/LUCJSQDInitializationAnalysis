#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
runfiles=$(find "${SCRIPT_DIR}/postprocess" -name "*.sh")
echo $SCRIPT_DIR
for i in $runfiles; do
    filename_no_extension=$(basename "${i%.*}")
    file2="${SCRIPT_DIR}/energies/${filename_no_extension}.txt"

    if [[ -f "$i" && -e "$file2" ]]; then
        file_month=$(stat -c "%y" "$file2" | cut -c6-7)

        if [[ "$file_month" == "10" ]]; then
	    echo "$file_month"
  	    cd ./postprocess
            sbatch "$i"
  	    cd ../
            echo "$i"
        fi
    elif [[ -f "$i" && ! -e "$file2" ]]; then
  	    cd ./postprocess
            sbatch "$i"
  	    cd ../
        echo "$i"
    fi
done
