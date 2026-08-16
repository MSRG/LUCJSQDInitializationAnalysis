#!/bin/bash
# Loop over files and only submit if the filename isn't already queued
for f in *StateVector*sh; do
    if ! grep -qxF "$f" queued_jobs.txt; then
        sbatch "$f"
    fi
done

squeue -u $USER -h -o "%j" > queued_jobs.txt
