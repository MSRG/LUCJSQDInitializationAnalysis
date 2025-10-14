
import psutil
from functools import partial
import sys
# !/Users/grierjones/miniconda3/envs/distributed_LUCJ/bin/python3.13 --version
# !/Users/grierjones/miniconda3/envs/distributed_LUCJ/bin/python3.13 -m pip install shap --upgrade 
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


from ansatzmap import get_zigzag_physical_layout

from tqdm import tqdm

from DDLUCJ import DDLUCJ, GrabAmps    

ampdict = GrabAmps("methane","aug-cc-pVDZ")

t1, t2 = ampdict["ML_exact"]
initDDLUCJ = DDLUCJ(StructurePath="../../../../classical/structures/methane50.xyz", 
                    BasisSet="aug-cc-pVDZ", 
                    NElec=int(10),
                    NOrb=int(9),
                    injected=True,
                    t1=t1, 
                    t2=t2,
                    n_reps = int(4),
                    optimization_level=3,
                    temp_dir="./",
                    clean_temp_dir=True,
                    n_jobs=64,                            
                    verbose=False)

counts = np.load(f"../counts/methane_LUCJ_L4_aug-cc-pVDZ_ML_exact.npz")
bitstrings = counts['bitstrings']
probarr = counts['probarr']
result_history, result = initDDLUCJ(postprocess=True,BitArray=bitstrings) 

dfcheck = energyDF[(energyDF['Molecule'] == name)&(energyDF['Basis Set']==basis)]
new_energy = result.energy + initDDLUCJ.nuclear_repulsion_energy
EnergyPath = f"../energies/methane_LUCJ_L4_aug-cc-pVDZ_ML_exact.txt" 
with open(EnergyPath,'w') as f:
    new_row={"Basis Set": basis, "Molecule": name, "Method": f"LUCJ(L=4)/ML_exact", "Energy": new_energy}
    for k,v in new_row.items():
        f.write(f"{v}\n")

