#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psutil
from functools import partial
import sys
# !{sys.executable} --version
# !{sys.executable} -m pip install shap --upgrade 
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
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from qiskit_addon_sqd.counts import bit_array_to_arrays


from ansatzmap import get_zigzag_physical_layout

from tqdm import tqdm

from DDLUCJ import DDLUCJ, GrabAmps


# In[2]:


BasisDirs=glob('data/*')


# In[3]:


energyDF=pd.read_csv("../../../classical/energies.csv",index_col=0)


# In[4]:


moldf = pd.read_csv('molecules.csv')
activespacedf = pd.read_csv("active_spaces.csv")


# In[ ]:





# In[ ]:





# In[5]:


BasisSets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']


# In[6]:


# os.mkdir('counts')
service = QiskitRuntimeService()


# In[7]:


# for i in tqdm(sorted(glob("./jobids/*txt")),desc='Running'):
#     i.split("/")[-1].replace('.txt','').split('_')
#     with open(i,'r') as f:
#         name,basis,k,L,JobID = [i.strip() for i in f.readlines()]
#     print(name,basis,k,L,JobID)

#     job = service.job(JobID)
#     primitive_result = job.result()
#     pub_result = primitive_result[0]
#     bit_array = pub_result.data.meas
#     bitstrings, probarr= bit_array_to_arrays(bit_array)

#     np.savez_compressed(f"./counts/{name}_LUCJ_L{L}_{basis}_{k}.npz", bitstrings=bitstrings,probarr=probarr)


# In[ ]:






# In[8]:


# os.mkdir('energies')


# In[9]:


postprocessed = []
for i in tqdm(sorted(glob("./jobids/*txt")),desc='Running'):
    i.split("/")[-1].replace('.txt','').split('_')
    with open(i,'r') as f:
        name,basis,k,L,JobID = [i.strip() for i in f.readlines()]
    print(name,basis,k,L,JobID)
    moldict = moldf[moldf['molecule']==name]

    n_electrons=moldict['n_electrons'].values[0]
    num_orbitals=moldict['num_orbitals'].values[0]
    xyzname = moldict['mol_filename'].values[0]
    pathxyz = os.path.join("../../../classical/structures/",xyzname)

    ampdict = GrabAmps(name,basis)

    t1, t2 = ampdict[k]

    JobPath = f"./jobids/{name}_LUCJ_L{L}_{basis}_{k}.txt" 
    EnergyPath = f"./energies/{name}_LUCJ_L{L}_{basis}_{k}.txt" 
    counts = np.load(f"./counts/{name}_LUCJ_L{L}_{basis}_{k}.npz")
    if os.path.exists(JobPath)==True and os.path.exists(EnergyPath)==False:
        print(f"Running {name}_LUCJ_L{L}_{basis}_{k}")
        initDDLUCJ = DDLUCJ(StructurePath=pathxyz, 
                            BasisSet=basis, 
                            NElec=n_electrons,
                            NOrb=num_orbitals,
                            injected=True,
                            t1=t1, 
                            t2=t2,
                            n_reps = int(L),
                            # channel = 'ibm_quantum_platform',
                            # instance = 'crn:v1:bluemix:public:quantum-computing:us-east:a/d2c50f33c43a44abb94280706332351d:21577587-df3e-4814-9e65-9c35f3e49ac9::',
                            # backend = "ibm_quebec",         
                            optimization_level=3,
                            temp_dir="./",
                            clean_temp_dir=True,
                            n_jobs=64,                            
                            verbose=False)

        bitstring_matrix_full, probs_arr_full = counts['bitstrings'], counts['probarr']
        bit_array = BitArray.from_bool_array(bitstring_matrix_full)
        result_history, result = initDDLUCJ(postprocess=True,BitArray=bit_array) 

        dfcheck = energyDF[(energyDF['Molecule'] == name)&(energyDF['Basis Set']==basis)]
        new_energy = result.energy + initDDLUCJ.nuclear_repulsion_energy
        with open(EnergyPath,'w') as f:
            new_row={"Basis Set": basis, "Molecule": name, "Method": f"LUCJ(L={L})/{k}", "Energy": new_energy}
            for k,v in new_row.items():
                f.write(f"{v}\n")



# In[ ]:




