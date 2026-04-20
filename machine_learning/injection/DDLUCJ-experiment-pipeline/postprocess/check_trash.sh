#!/bin/bash

# Define your search patterns in an array for better maintenance
patterns=(
    "(Z)-1-fluoroprop-1-ene_LUCJ_L2_aug-cc-pVDZ_ML"
    "buta-1,3-diene_LUCJ_L1_aug-cc-pVDZ_CCSD"
    "fluoroform_LUCJ_L1_STO-3G_CCSD"
    "fluoroform_LUCJ_L1_aug-cc-pVDZ_ML_exact"
    "fluoroform_LUCJ_L1_cc-pVDZ_ML_exact"
)

for pattern in "${patterns[@]}"; do
    echo "--- Checking: $pattern ---"
    
    # 1. grep -l returns the filename only (e.g., job.o12302110)
    # 2. sed 's/job.o//' removes the prefix to leave just the ID
    file=$(grep -il "$pattern" job.* | head -n 1)
    
    if [ -n "$file" ]; then
        job_id=$(echo "$file" | sed 's/job.o//')
        echo "Running seff for Job ID: $job_id"
        seff "$job_id"
    else
        echo "No job file found for: $pattern"
    fi
    echo ""
done
