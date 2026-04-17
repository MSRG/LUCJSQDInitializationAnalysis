#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
# ! export CMAKE_PREFIX_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/symengine/0.14.0"
# sys.path.insert(0, "/project/6006115/gjones/molrep/lib/python3.10/site-packages")
# !{sys.executable} -m pip uninstall numpy --yes
# !{sys.executable} -m pip install --no-index --upgrade pip
# !{sys.executable} -m pip install -e /home/gjones/projects/def-jacobsen/gjones/qiskit-addon-dice-solver/
# !{sys.executable} -m pip install numpy==1.26.4
# !{sys.executable} -m pip install -e /scratch/gjones/distributed_LUCJ/
# !pip install -e /home/gjones/projects/def-jacobsen/gjones/qiskit-addon-dice-solver/
# !pip install -e /scratch/gjones/distributed_LUCJ/
# !{sys.executable} -m pip install pandas
import psutil
from functools import partial

import joblib
import time
from shutil import copy
import numpy as np
import pandas as pd
#import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
# import xgboost as xgb

from glob import glob
# import psi4
# from helper_CC_ML_spacial import *

import pyscf
from pyscf import gto, scf, mcscf, cc

import ffsim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from qiskit_addon_sqd.counts import bit_array_to_arrays
from qiskit.primitives import StatevectorSampler, BitArray


from ansatzmap import get_zigzag_physical_layout

from tqdm import tqdm

from DDLUCJ import DDLUCJ, GrabAmps    


# In[ ]:


BasisDirs=glob('data/*')


# In[ ]:


energyDF=pd.read_csv("../../../classical/energies.csv",index_col=0)


# In[ ]:


moldf = pd.read_csv('molecules.csv')
activespacedf = pd.read_csv("active_spaces.csv")


# In[ ]:


BasisSets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']


# In[ ]:


trash = pd.read_excel("trash.xlsx")[['Basis Set','Molecule','L','Injected']]


# In[ ]:


n_jobs = os.environ.get("SLURM_NTASKS")
print("SLURM_NTASKS",n_jobs)


# In[ ]:


for _, row in trash.iterrows():
    basis, name, layer, injection = row.to_list()
    filename = f"{name}_LUCJ_L{layer}_{basis}_{injection}.sh"

    moldict = moldf[moldf['molecule']==name]
    if moldict.empty:
        print(f"MISSING IN MOLECULES.CSV: {name}")
        continue
    n_electrons=moldict['n_electrons'].values[0]
    num_orbitals=moldict['num_orbitals'].values[0]
    xyzname = moldict['mol_filename'].values[0]
    pathxyz = os.path.join("../../../classical/structures/",xyzname)

    JobPath = f"./jobids/{name}_LUCJ_L{layer}_{basis}_{injection}.txt" 
    EnergyPath = f"./energies/{name}_LUCJ_L{layer}_{basis}_{injection}.txt" 


    print(f"Running {name}_LUCJ_L{layer}_{basis}_{injection}")
    # run(pathxyz,name,basis,n_electrons,num_orbitals,layer,injection)
    ampdict = GrabAmps(name,basis)

    t1, t2 = ampdict[injection]
    initDDLUCJ = DDLUCJ(StructurePath=pathxyz, 
                        BasisSet=basis, 
                        NElec=int(n_electrons),
                        NOrb=int(num_orbitals),
                        injected=True,
                        t1=t1, 
                        t2=t2,
                        n_reps = int(layer),
                        optimization_level=3,
                        temp_dir="./",
                        clean_temp_dir=False,
                        n_jobs=n_jobs,
                        num_batches = 10,
                        max_iterations=5,
                        samples_per_batch=1000,
                        verbose=True)

    counts = np.load(f"./counts/{name}_LUCJ_L{layer}_{basis}_{injection}.npz")
    bitstrings = counts['bitstrings']
    probarr = counts['probarr']
    bitstrings = BitArray.from_bool_array(bitstrings)

    result_history, result = initDDLUCJ(postprocess=True,BitArray=bitstrings) 

    new_energy = result.energy + initDDLUCJ.nuclear_repulsion_energy
    EnergyPath = f"./energies/{name}_LUCJ_L{layer}_{basis}_{injection}.txt" 
    with open(EnergyPath,'w') as f:
        new_row={{"Basis Set": "{basis}", "Molecule": "{name}", "Method": f"LUCJ(L={layer})/{injection}", "Energy": new_energy}}
        for k,v in new_row.items():
            f.write(f"{{v}}\\n")





# In[ ]:




