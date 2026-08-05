#!/bin/bash
files=$(find . -type f -name "dmrg*py")
topdir=$(pwd)

for i in $files; do
    dirpath=$(dirname "$i")
    cp "${topdir}/classical.py" "${dirpath}/classical.py"

    cd "$dirpath" || { echo "Cannot cd to $dirpath"; continue; }

    echo "Submitting from $PWD"
    sbatch --export=ALL dmrg_run.sh

    cd "$topdir" || exit
done

