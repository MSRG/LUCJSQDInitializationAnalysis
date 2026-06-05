#!/bin/bash

files=(
    "but-1-yne_LUCJ_L5_aug-cc-pVDZ_random.txt"
    "buta-1,3-diene_LUCJ_L1_aug-cc-pVDZ_random.txt"
    "fluoroform_LUCJ_L1_aug-cc-pVDZ_ML_exact.txt"
    "prop-2-en-1-ol_LUCJ_L3_STO-3G_MP2.txt"
    "buta-1,3-diene_LUCJ_L3_aug-cc-pVDZ_CCSD.txt"
    "prop-2-en-1-ol_LUCJ_L2_STO-3G_random.txt"
    "fluoroform_LUCJ_L1_cc-pVDZ_ML_exact.txt"
    "prop-2-en-1-ol_LUCJ_L5_aug-cc-pVDZ_random.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L2_aug-cc-pVDZ_ML.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L4_STO-3G_ML.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L3_aug-cc-pVDZ_ML.txt"
    "prop-2-en-1-ol_LUCJ_L5_aug-cc-pVDZ_ML_exact.txt"
    "prop-2-en-1-ol_LUCJ_L3_cc-pVDZ_ML.txt"
    "prop-2-en-1-ol_LUCJ_L5_cc-pVDZ_ML_exact.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L3_aug-cc-pVDZ_random.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L4_aug-cc-pVDZ_random.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L1_aug-cc-pVDZ_ML_exact.txt"
    "fluoroform_LUCJ_L1_STO-3G_CCSD.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L4_cc-pVDZ_ML.txt"
    "but-1-yne_LUCJ_L3_cc-pVDZ_MP2.txt"
    "prop-2-en-1-ol_LUCJ_L2_STO-3G_ML.txt"
    "buta-1,3-diene_LUCJ_L1_aug-cc-pVDZ_CCSD.txt"
    "(Z)-1-fluoroprop-1-ene_LUCJ_L2_cc-pVDZ_CCSD.txt"
)

for f in "${files[@]}"; do
    cp "$f" "${f}.bak"
done
