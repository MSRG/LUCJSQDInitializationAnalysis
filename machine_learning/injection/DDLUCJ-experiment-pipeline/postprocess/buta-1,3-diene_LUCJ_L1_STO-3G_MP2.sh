#!/bin/bash
#SBATCH --time=0-24:00:00
#SBATCH -J buta-1,3-diene_LUCJ_L1_STO-3G_MP2
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=2GB
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
pip install -e /scratch/gjones/distributed_LUCJ/

export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running in directory: $(pwd)"
echo "buta-1,3-diene_LUCJ_L1_STO-3G_MP2"
python "buta-1,3-diene_LUCJ_L1_STO-3G_MP2.py"
echo "File run"    
