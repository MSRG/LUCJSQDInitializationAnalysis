#!/bin/bash
#SBATCH --time=1-0:00:00
#SBATCH -J fluoroform_LUCJ_L5_cc-pVDZ_ML_exact
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=1000M
#SBATCH --error=job.e%J
#SBATCH --output=job.o%j



echo 'About to run python file'
echo 'About to run python file'
module load python/3.10

module load StdEnv/2023
module load openmpi
module load symengine rust
module load hdf5
module load openblas

echo "TEMP DIR: $SLURM_TMPDIR"
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -e /home/gjones/projects/def-jacobsen/gjones/qiskit-addon-dice-solver/


export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running in directory: $(pwd)"

python fluoroform_LUCJ_L5_cc-pVDZ_ML_exact.py 
echo "File run"    
