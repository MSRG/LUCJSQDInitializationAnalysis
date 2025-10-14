#!/bin/bash
#SBATCH --time=0-8:00:00
#SBATCH -J but-1-yne_LUCJ_L4_STO-3G_random
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=1000M
#SBATCH --error=job.e%J
#SBATCH --output=job.o%j



echo 'About to run python file'
module load python/3.10
module load openmpi
module load symengine rust
module load hdf5
source /lustre09/project/6004825/gjones/ENV/bin/activate

echo "Running in directory: $(pwd)"
python but-1-yne_LUCJ_L4_STO-3G_random.py 
echo "File run"    
