#!/bin/bash

# Define the output CSV file and header
OUTPUT_FILE="job_stats.csv"
echo "JobID,System_Name,Error_Type,State,CPU_Efficiency,Mem_Efficiency,Wallclock" > $OUTPUT_FILE

# Loop through all job output files
for file in job.o*; do
    # Extract JobID from the filename
    jobid=${file#job.o}

    # 1. Get stats from seff
    seff_out=$(seff $jobid 2>/dev/null)
    if [ $? -eq 0 ]; then
        state=$(echo "$seff_out" | grep "State:" | awk '{print $2}')
        cpu_eff=$(echo "$seff_out" | grep "CPU Efficiency:" | awk '{print $3}')
        mem_eff=$(echo "$seff_out" | grep "Memory Efficiency:" | awk '{print $3}')
        wallclock=$(echo "$seff_out" | grep "Wall-clock time:" | awk '{print $3}')
    else
        state="N/A"; cpu_eff="N/A"; mem_eff="N/A"; wallclock="N/A"
    fi

    # 2. Parse File Content
    # Extract System Name (e.g., but-1-yne_LUCJ_L5_aug-cc-pVDZ_random)
    # This looks for the line after "Running in directory:"
    system_name=$(grep -A 1 "Running in directory:" "$file" | tail -n 1 | tr -d '\r')

    # If system_name is empty, try to find a .py file mention as a fallback
    if [ -z "$system_name" ]; then
        system_name=$(grep -oP 'python "\K[^"]+' "$file" | head -n 1)
    fi

    # Identify the error type
    if grep -q "oom_kill" "$file"; then
        error_msg="OOM_Killed"
    elif grep -q "Invalid value for environment variable OMP_NUM_THREADS" "$file"; then
        error_msg="OMP_THREADS_Error"
    elif grep -q "Killed" "$file"; then
        error_msg="Killed"
    elif grep -q "Successfully installed" "$file" && ! grep -q "Iteration" "$file"; then
        error_msg="Setup_Only/Incomplete"
    else
        error_msg="None"
    fi

    # 3. Append to CSV
    echo "$jobid,$system_name,$error_msg,$state,$cpu_eff,$mem_eff,$wallclock" >> $OUTPUT_FILE
done

echo "Parsing complete. Results saved to $OUTPUT_FILE"
