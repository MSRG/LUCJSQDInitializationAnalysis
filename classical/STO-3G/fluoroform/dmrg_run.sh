#!/bin/bash
#SBATCH --time=0-2:00:00
#SBATCH --account=rrg-jacobsen-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5GB
#SBATCH --job-name="DMRG/STO-3G/fluoroform"
#SBATCH --error=job.e%J
#SBATCH --output=job.o%J

echo 'About to run python file'

# Load modules in correct order — StdEnv first
module load StdEnv/2023
module load python/3.11
module load openmpi
module load symengine rust
module load hdf5
module load openblas

echo "TEMP DIR: $SLURM_TMPDIR"

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install -e /home/gjones/projects/def-jacobsen/gjones/qiskit-addon-dice-solver/
pip install -e /home/gjones/scratch/distributed_LUCJ/
pip install git+https://github.com/pyscf/dmrgscf

PYSCFHOME=$(python -c "import pyscf; import os; print(os.path.dirname(pyscf.__file__))")
echo "PySCF home: $PYSCFHOME"

wget https://raw.githubusercontent.com/pyscf/dmrgscf/master/pyscf/dmrgscf/settings.py.example
mv settings.py.example ${PYSCFHOME}/dmrgscf/settings.py
chmod +x ${PYSCFHOME}/dmrgscf/nevpt2_mpi.py

pip install 'block2==0.5.3'

export LD_LIBRARY_PATH=$EBROOTOPENBLAS/lib:$LD_LIBRARY_PATH
export PATH=$SLURM_TMPDIR/env/bin:$PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export LD_LIBRARY_PATH=$SLURM_TMPDIR/env/lib:$LD_LIBRARY_PATH

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Running in directory: $(pwd)"
echo "Using input file: dmrg_input.py"
python dmrg_input.py
echo "File run complete for DMRG/STO-3G/fluoroform."
