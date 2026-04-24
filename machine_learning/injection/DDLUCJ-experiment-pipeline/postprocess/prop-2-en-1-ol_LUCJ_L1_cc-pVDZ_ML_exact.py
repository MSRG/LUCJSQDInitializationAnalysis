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

ampdict = GrabAmps("prop-2-en-1-ol","cc-pVDZ")

t1, t2 = ampdict["ML_exact"]
initDDLUCJ = DDLUCJ(StructurePath="../../../../classical/structures/GDB04_33.xyz", 
                    BasisSet="cc-pVDZ", 
                    NElec=int(32),
                    NOrb=int(26),
                    injected=True,
                    t1=t1, 
                    t2=t2,
                    n_reps = int(1),
                    optimization_level=3,
                    temp_dir="./",
                    clean_temp_dir=True,
                    n_jobs=32,
                    num_batches = 10,
                    max_iterations=5,
                    samples_per_batch=1000,
                    verbose=False)

counts = np.load(f"../counts/prop-2-en-1-ol_LUCJ_L1_cc-pVDZ_ML_exact.npz")
bitstrings = counts['bitstrings']
bitstrings = BitArray.from_bool_array(bitstrings)

new_energy, subspace = initDDLUCJ(postprocess=True,BitArray=bitstrings,usefulqrum=True,bitarraypath=f"../counts/prop-2-en-1-ol_LUCJ_L1_cc-pVDZ_ML_exact.npz") 

EnergyPath = f"../energies/prop-2-en-1-ol_LUCJ_L1_cc-pVDZ_ML_exact.txt" 

with open(EnergyPath,'w') as f:
    new_row={"Basis Set": "cc-pVDZ", "Molecule": "prop-2-en-1-ol", "Method": f"LUCJ(L=1)/ML_exact", "Energy": new_energy}
    for k,v in new_row.items():
        f.write(f"{v}\n")
