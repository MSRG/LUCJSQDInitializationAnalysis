#!/bin/bash
sbatch "fluoroform_LUCJ_L1_aug-cc-pVDZ_ML_exact.sh"
sbatch "fluoroform_LUCJ_L1_cc-pVDZ_ML_exact.sh"
sbatch "(Z)-1-fluoroprop-1-ene_LUCJ_L2_aug-cc-pVDZ_ML.sh"
sbatch "prop-2-en-1-ol_LUCJ_L5_aug-cc-pVDZ_ML_exact.sh"
sbatch "(Z)-1-fluoroprop-1-ene_LUCJ_L3_aug-cc-pVDZ_random.sh"
sbatch "fluoroform_LUCJ_L1_STO-3G_CCSD.sh"
sbatch "prop-2-en-1-ol_LUCJ_L2_STO-3G_ML.sh"
sbatch "buta-1,3-diene_LUCJ_L1_aug-cc-pVDZ_CCSD.sh"
