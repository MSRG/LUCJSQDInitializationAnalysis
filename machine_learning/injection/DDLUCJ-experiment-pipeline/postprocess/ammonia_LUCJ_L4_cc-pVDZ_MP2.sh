#!/bin/bash
#SBATCH --time=0-8:00:00
#SBATCH -J ammonia_LUCJ_L4_cc-pVDZ_MP2
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=5000M
#SBATCH --error=job.e%J
#SBATCH --output=job.o%j



echo 'About to run python file'
module load python/3.10
module load openmpi
module load symengine rust
module load hdf5
module load openblas
export UCX_VFS_ENABLE=no
source /lustre09/project/6004825/gjones/ENV/bin/activate
export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH
echo "Running in directory: $(pwd)"
python ammonia_LUCJ_L4_cc-pVDZ_MP2.py 
echo "File run"    
