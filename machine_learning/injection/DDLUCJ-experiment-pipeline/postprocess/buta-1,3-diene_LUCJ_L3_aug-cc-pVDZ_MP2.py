import psutil
from functools import partial
import sys
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
from glob import glob
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

ampdict = GrabAmps("buta-1,3-diene","aug-cc-pVDZ")

t1, t2 = ampdict["MP2"]
initDDLUCJ = DDLUCJ(StructurePath="../../../../classical/structures/GDB04_53.xyz", 
                    BasisSet="aug-cc-pVDZ", 
                    NElec=int(30),
                    NOrb=int(26),
                    injected=True,
                    t1=t1, 
                    t2=t2,
                    n_reps = int(3),
                    optimization_level=3,
                    temp_dir="./",
                    clean_temp_dir=True,
                    n_jobs=32,
                    num_batches = 10,
                    max_iterations=5,
                    samples_per_batch=1000,
                    verbose=False)

counts = np.load(f"../counts/buta-1,3-diene_LUCJ_L3_aug-cc-pVDZ_MP2.npz")
bitstrings = counts['bitstrings']
bitstrings = BitArray.from_bool_array(bitstrings)

new_energy, subspace = initDDLUCJ(postprocess=True,BitArray=bitstrings,usefulqrum=True,bitarraypath=f"../counts/buta-1,3-diene_LUCJ_L3_aug-cc-pVDZ_MP2.npz") 

EnergyPath = f"../energies/buta-1,3-diene_LUCJ_L3_aug-cc-pVDZ_MP2.txt" 

with open(EnergyPath,'w') as f:
    new_row={"Basis Set": "aug-cc-pVDZ", "Molecule": "buta-1,3-diene", "Method": f"LUCJ(L=3)/MP2", "Energy": new_energy}
    for k,v in new_row.items():
        f.write(f"{v}\n")
