import psutil
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
import xgboost as xgb
import json
from glob import glob
import psi4
from helper_CC_ML_spacial import *

import pyscf
import pyscf.cc
import pyscf.mcscf
import ffsim
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

home = os.path.expanduser("~")
molecules_dir = os.path.join(home, "DDLUCJ", "machine_learning", "injection","sqd-experiment-pipeline", "molecules.csv")
top5 = ['doublecheck', 't2start', 't2mag', 'orbdiff', 'diag']


# Load in XGBoost Model
basis_sets = ['STO-3G','cc-pVDZ','aug-cc-pVDZ']
basis = basis_sets[2]
model_name = f'optimised_{basis}_best_model_100.pkl'
model = joblib.load(os.path.join(os.path.expanduser('~'),'DDLUCJ', 'machine_learning', 'out', model_name))

# Retrieve molecule filenames, formulas, and active spaces in minimal basis set
molecules = pd.read_csv(molecules_dir)

for molecule in molecules['molecule']:

    mol_row = molecules[molecules['molecule'] == molecule].iloc[0]
    mol_file =  mol_row['mol_filename']

    # Load Molecule in Python and prepare Helper Object for CCSD calculations
    with open(os.path.join(os.path.expanduser('~'),'DDLUCJ', 'machine_learning', 'data', mol_file),'r') as f:
        text=f.read()

    mol = psi4.geometry(text)
    psi4.core.clean()
    psi4.core.be_quiet()

    psi4.set_options({'basis': basis,
                    'scf_type':     'pk',
                    'reference':    'rohf',
                    'mp2_type':     'conv',
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8})

    rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
    scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

    A=HelperCCEnergy(mol, rhf_e, scf_wfn,freeze_core=False)

    # Generate 6 pairs of initialization types

    # t1_zeroes and t2_zeroes
    start = time.time()
    t1_zeroes, t2_zeroes = np.zeros_like(A.t1), np.zeros_like(A.t2start)
    end = time.time()
    elapsed_zeroes = end - start

    # t1_zeroes and t2_MP2
    start = time.time()
    t2_MP2 = A.t2start
    end = time.time()
    elapsed_MP2 = end - start

    # t1_rand and t2_rand
    start = time.time()
    t1_rand, t2_rand = np.random.uniform(low=-100, high=100, size=A.t1.shape), np.random.uniform(low=-100, high=100, size=A.t2start.shape)
    end = time.time()
    elapsed_rand = end - start

    # t1_exact and t2_exact standard (Note that I only call t2 when I want it to be exact to avoid potential aliasing)
    start = time.time()
    A.compute_energy()
    end = time.time()
    elapsed_exact = end - start
    t1_exact = A.t1
    t2_exact = A.t2

    # t1_zeroes and t2_predicted
    start = time.time()
    X= np.vstack([getattr(A,i).flatten() for i in top5]).T
    y = model.predict(X)
    t2_ML = y.reshape(*A.t2start.shape)
    end = time.time()
    elapsed_ML = end - start

    # t1_exact and t2_exact from ML guess
    A.t1 = t1_zeroes
    A.t2 = t2_ML
    A.compute_energy()
    end = time.time()
    elapsed_ML_exact = end - start
    t1_ML_exact = A.t1
    t2_ML_exact = A.t2

    # Save them under data/amplitudes for later use
    # Prepare directory
    amp_dir = os.path.join(home, "DDLUCJ", "machine_learning", "injection", "sqd-experiment-pipeline", "data", basis, "amplitudes")
    os.makedirs(amp_dir, exist_ok=True)

    # Save arrays
    np.savez(os.path.join(amp_dir, f"amps_{mol_row['molecule']}_{basis}.npz"),
            t1_zeroes=t1_zeroes, t2_zeroes=t2_zeroes,
            t1_rand=t1_rand, t2_rand=t2_rand,
            t1_exact=t1_exact, t2_exact=t2_exact,
            t2_MP2=t2_MP2, t2_ML=t2_ML,
            t1_ML_exact=t1_ML_exact, t2_ML_exact=t2_ML_exact)

    # Save meta data
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Calculate metrics (example for t2 arrays)
    metrics = {}
    for name, arr in [('t2_zeroes', t2_zeroes), ('t2_rand', t2_rand), ('t2_MP2', t2_MP2), ('t2_ML', t2_ML), ('t2_ML_exact', t2_ML_exact)]:
        metrics[name] = {
            'MAE': float(mean_absolute_error(t2_exact.flatten(), arr.flatten())),
            'MSE': float(mean_squared_error(t2_exact.flatten(), arr.flatten())),
            'MAPE': float(np.mean(np.abs((t2_exact.flatten() - arr.flatten()) / (t2_exact.flatten() + 1e-8))))
        }

    # Save times and metrics
    metadata = {
        'molecule': mol_row['molecule'],
        'basis': basis,
        'amplitude_generation_times': {
            'Zeroes': elapsed_zeroes,
            'Random': elapsed_rand,
            'MP2*': elapsed_MP2,
            'Exact': elapsed_exact,
            'ML': elapsed_ML,
            'ML_exact': elapsed_ML_exact
        },
        'metrics': metrics
    }

    with open(os.path.join(amp_dir, f"amp_metrics_{mol_row['molecule']}_{basis}.json"), 'w') as f:
        json.dump(metadata, f, indent=2)