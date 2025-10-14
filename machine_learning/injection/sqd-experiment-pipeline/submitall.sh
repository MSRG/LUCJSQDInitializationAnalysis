#!/bin/bash

runfiles=$(find ./postprocess -name "*sh")

for i in $runfiles; do
 filename_no_extension=$(basename "${i%.*}")
 file2="./energies/${filename_no_extension}.txt"
 if [[ -e "$i" && ! -e "$file2" ]]; then
  cd ./postprocess
  sbatch $(basename ${i})
  cd ../
 fi
done
