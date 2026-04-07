
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
from qiskit.primitives import StatevectorSampler, BitArray


from ansatzmap import get_zigzag_physical_layout

from tqdm import tqdm

from DDLUCJ import DDLUCJ, GrabAmps    

ampdict = GrabAmps("fluoroform","STO-3G")

t1, t2 = ampdict["MP2"]
initDDLUCJ = DDLUCJ(StructurePath="../../../../classical/structures/GDB04_5.xyz", 
                    BasisSet="STO-3G", 
                    NElec=int(34),
                    NOrb=int(21),
                    injected=True,
                    t1=t1, 
                    t2=t2,
                    n_reps = int(1),
                    optimization_level=3,
                    temp_dir="./",
                    clean_temp_dir=True,
                    n_jobs=64,
                    num_batches = 10,
                    max_iterations=5,
                    samples_per_batch=1000,
                    verbose=False)

counts = np.load(f"../counts/fluoroform_LUCJ_L1_STO-3G_MP2.npz")
bitstrings = counts['bitstrings']
probarr = counts['probarr']
bitstrings = BitArray.from_bool_array(bitstrings)

result_history, result = initDDLUCJ(postprocess=True,BitArray=bitstrings) 

new_energy = result.energy + initDDLUCJ.nuclear_repulsion_energy
EnergyPath = f"../energies/fluoroform_LUCJ_L1_STO-3G_MP2.txt" 
with open(EnergyPath,'w') as f:
    new_row={"Basis Set": "STO-3G", "Molecule": "fluoroform", "Method": f"LUCJ(L=1)/MP2", "Energy": new_energy}
    for k,v in new_row.items():
        f.write(f"{v}\n")

