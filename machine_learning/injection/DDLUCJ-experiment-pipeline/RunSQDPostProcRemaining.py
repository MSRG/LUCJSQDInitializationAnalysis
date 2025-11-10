#!/usr/bin/env python
# coding: utf-8

# <span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [1]</a>'.</span>

# <span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>

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
from qiskit.primitives import StatevectorSampler, BitArray
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from qiskit_addon_sqd.counts import bit_array_to_arrays


from ansatzmap import get_zigzag_physical_layout

from tqdm import tqdm

from DDLUCJ import DDLUCJ, GrabAmps


# In[ ]:


repostprocess = pd.read_excel("repostprocess.xlsx",index_col=0).reset_index(drop=True)


# In[ ]:





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


# os.mkdir('counts')
service = QiskitRuntimeService()


# In[ ]:


"../../../classical/structures/"


# In[ ]:


def PostProcessLocally(pathxyz,name,basis,n_electrons,num_orbitals,L,k):

    ampdict = GrabAmps(name,basis)
    
    t1, t2 = ampdict[k]
    initDDLUCJ = DDLUCJ(StructurePath=pathxyz, 
                        BasisSet=basis, 
                        NElec=n_electrons,
                        NOrb=num_orbitals,
                        injected=True,
                        t1=t1, 
                        t2=t2,
                        n_reps = L,
                        optimization_level=3,
                        n_jobs=None,                            
                        verbose=False)
    
    counts = np.load(f"./counts/{name}_LUCJ_L{L}_{basis}_{k}.npz")
    bitstrings = counts['bitstrings']
    probarr = counts['probarr']
    bitstrings = BitArray.from_bool_array(bitstrings)
    
    result_history, result = initDDLUCJ(postprocess=True,BitArray=bitstrings) 
    
    new_energy = result.energy + initDDLUCJ.nuclear_repulsion_energy
    EnergyPath = f"./energies/{name}_LUCJ_L{L}_{basis}_{k}.txt" 
    with open(EnergyPath,'w') as f:
        new_row={{"Basis Set": "{basis}", "Molecule": "{name}", "Method": f"LUCJ(L={L})/{k}", "Energy": new_energy}}
        for k,v in new_row.items():
            f.write(f"{v}\\n")


# In[ ]:


postprocessed = []
submoldf = moldf[['n_electrons','num_orbitals','mol_filename','molecule']]
for _, row in tqdm(repostprocess.iterrows(),desc='Running'):
    rowdict = row.to_dict()
    basis = rowdict['Basis Set']
    name = rowdict['Molecule']
    L = rowdict['L']
    k = rowdict['Injected']
    n_electrons, num_orbitals, xyzname = submoldf.loc[submoldf['molecule'] == name,['n_electrons','num_orbitals','mol_filename']].values[0]
    print(n_electrons, num_orbitals, xyzname)
    pathxyz = os.path.join("../../../classical/structures/",xyzname)    
    print(pathxyz)
    
    JobPath = f"./jobids/{name}_LUCJ_L{L}_{basis}_{k}.txt" 
    EnergyPath = f"./energies/{name}_LUCJ_L{L}_{basis}_{k}.txt"     

    if os.path.exists(JobPath) and os.path.exists(pathxyz):
        print(f"Running {name}_LUCJ_L{L}_{basis}_{k}")
        # print(name,basis,k,L,JobID)
        PostProcessLocally(pathxyz,name,basis,n_electrons,num_orbitals,L,k)    


# In[ ]:




