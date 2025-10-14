#!/bin/bash
#SBATCH --time=4-0:00:00
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=5000M
#SBATCH --error=job.e%J
#sbatch --output=job.o%j
#SBATCH -J postprocess


echo 'About to run python file'
module load python/3.10
module load openmpi
module load symengine rust
module load hdf5
source /lustre09/project/6004825/gjones/ENV/bin/activate

echo "Running in directory: $(pwd)"
echo "Using input file: input.py"

python RunSQDPostProc.py 

echo "File run complete for $NAME."

