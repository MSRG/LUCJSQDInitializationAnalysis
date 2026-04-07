#!/bin/bash
#SBATCH --time=1-0:00:00
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=1000M
#SBATCH --job-name=methane_LUCJ_L5_aug-cc-pVDZ_ML
#SBATCH --error=job.e%J
#SBATCH --output=job.o%j



echo 'About to run python file'
module load python/3.10
module load openmpi
module load symengine rust
module load hdf5
module load openblas
source /lustre09/project/6004825/gjones/ENV/bin/activate
export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running in directory: $(pwd)"

python methane_LUCJ_L5_aug-cc-pVDZ_ML.py 
echo "File run"    
