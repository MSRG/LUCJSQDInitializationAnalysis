#!/bin/bash
#SBATCH --time=0-8:00:00
#SBATCH -J but-1-yne_LUCJ_L2_cc-pVDZ_CCSD_StateVector_Shots10000
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5GB
#SBATCH --error=job.e%J
#SBATCH --output=job.o%j

set -e

echo 'About to run python file'
module load python/3.11
module load StdEnv/2023
module load openmpi
module load symengine rust
module load hdf5
module load openblas

echo "TEMP DIR: $SLURM_TMPDIR"
source /home/gjones/scratch/tmp_env/bin/activate
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --find-links=/home/gjones/projects/def-jacobsen/gjones/wheels/ fulqrum
# pip install -e /home/gjones/scratch/distributed_LUCJ/
# pip install openfermion

export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running in directory: $(pwd)"
echo "but-1-yne_LUCJ_L2_cc-pVDZ_CCSD_StateVector_Shots10000"
python "but-1-yne_LUCJ_L2_cc-pVDZ_CCSD_StateVector_Shots10000.py"
echo "File run"
