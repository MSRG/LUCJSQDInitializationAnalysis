import numpy as np
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
import pandas as pd
import pyscf
import math
from pyscf import gto, scf, mcscf, cc, tools, lib, dmrgscf


def run_DMRG(structure,
             basis,
             n_electrons,
             num_orbitals,
             sym,
             spin_sq,
             charge,
             n_jobs = None
            ):
    dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
    dmrgscf.settings.MPIPREFIX = ''

    print(structure,basis,sym,spin_sq,charge)
    mol = gto.Mole()
    mol.build(
        atom=structure,
        basis=basis,
        symmetry=sym,
        spin=spin_sq,
        charge=charge
    )
    mol.verbose = 0
    s = spin_sq / 2
    total_spin = s * (s + 1)  
    
    if spin_sq==0:
        open_shell=False
    else:
        open_shell=True
    
    print(f"2S {spin_sq}")   
    print(f"S * (S+1) {total_spin}") 
    
    if open_shell:
        HF = scf.UHF(mol).run()
    else:
        if np.sum(mol.nelec)<=30:
            HF = scf.RHF(mol).run()
        else:
            # Increase cycles and remove linear dependence
            mol.max_cycle = 500
            HF = scf.RHF(mol).apply(scf.addons.remove_linear_dep_).run()
    
    
     

    
    # Determine the number of alpha and beta electrons
    num_elec_a = (n_electrons + mol.spin) // 2
    num_elec_b = (n_electrons - mol.spin) // 2    
    nelec = (num_elec_a,num_elec_b)
    # Load the OpenMP library (usually libomp on macOS/brew)
#   try:
#       omp = ctypes.CDLL("libomp.dylib")
#   except OSError:
#       # Fallback for different installations
#       omp = ctypes.CDLL("libgomp.dylib")
    
    # Call the function to get max threads
    #num_threads = omp.omp_get_max_threads()
    # print(f"OMP Max Threads: {num_threads}")
    mc = mcscf.CASCI(HF, num_orbitals, nelec)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=1000, tol=1E-10)
    mc.fcisolver.runtimeDir = lib.param.TMPDIR
    mc.fcisolver.scratchDirectory = lib.param.TMPDIR
    # Derive available memory from SLURM allocation (in GB), with fallback
    slurm_mem_per_cpu_mb = int(os.environ.get("SLURM_MEM_PER_CPU", 4000))  # MB
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    total_mem_gb = (slurm_mem_per_cpu_mb * slurm_cpus) // 1000
    
    # Leave ~10% headroom for PySCF, OS, etc.
    block2_mem_gb = int(total_mem_gb * 0.85)
    
    mc.fcisolver.mpiprefix = ''   # ADD THIS — prevent nested srun
    mc.fcisolver.memory = block2_mem_gb
    mc.fcisolver.extraline = ['symmetrize_ints 1e-6']  # relax from 1e-10; safe for D2h subgroup
    mc.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 1))  # default prevents TypeError
    
    mc.canonicalization = True
    mc.natorb = True
    total_energy = mc.kernel()[0]
    return total_energy    
