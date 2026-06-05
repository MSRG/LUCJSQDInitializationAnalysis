#!/bin/bash

files=(
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/(Z)-1-fluoroprop-1-ene_LUCJ_L2_aug-cc-pVDZ_ML.txt"
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/(Z)-1-fluoroprop-1-ene_LUCJ_L3_aug-cc-pVDZ_random.txt"
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/buta-1,3-diene_LUCJ_L1_aug-cc-pVDZ_CCSD.txt"
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/fluoroform_LUCJ_L1_STO-3G_CCSD.txt"
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/fluoroform_LUCJ_L1_aug-cc-pVDZ_ML_exact.txt"
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/fluoroform_LUCJ_L1_cc-pVDZ_ML_exact.txt"
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/prop-2-en-1-ol_LUCJ_L2_STO-3G_ML.txt"
  "machine_learning/injection/DDLUCJ-experiment-pipeline/energies/prop-2-en-1-ol_LUCJ_L5_aug-cc-pVDZ_ML_exact.txt"
)

echo "Files WITHOUT energy < -1000:"
for f in "${files[@]}"; do
  energy=$(tail -1 "$f")
  # Use awk to check if the value is >= -1000 (i.e., not sufficiently negative)
  if awk -v e="$energy" 'BEGIN { exit !(e > -1000) }'; then
    echo "  $f  -->  $energy"
  else

    echo "BAD  $f  -->  $energy"
  fi
done
