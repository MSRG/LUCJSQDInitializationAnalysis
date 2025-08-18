import psutil
import sys
# !{sys.executable} --version
# !{sys.executable} -m pip install shap --upgrade 
import joblib
import time
from shutil import copy
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob
import psi4
from helper_CC_ML_spacial import *

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
    

# Most important features (top 5 by SHAP):
# doublecheck: Numerator of the MP2 t2-amplitude, two-electron integral <ik || ab>
# t2start: Initial MP2 t2-amplitude
# t2mag: Magnitude of the MP2 t2-amplitude
# orbdiff: Denominator of the MP2 t2-amplitude
# diag: Binary feature denoting whether a=b (virtual orbits are the same)
top5 = ['doublecheck', 't2start', 't2mag', 'orbdiff', 'diag']

# 31 Features order in X, including t2
properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']


basis_sets = ['STO-3G', 'cc-pVDZ', 'aug-cc-pVDZ']

with open("out/train_names.txt", "r") as f:
    lines = f.readlines()

train = [element[:-1] for element in lines]

with open("out/test_names.txt", "r") as f:
    lines = f.readlines()

test = [element[:-1] for element in lines]


X_test_all = {}
y_test_all = {}

for basis in basis_sets:
    filenames = test.copy()
    t1 = time.time()

    data_dict = {}
    for fn in filenames:
        struct = os.path.basename(fn)
        
        with open(fn,'r') as f:
            text=f.read()
        
        mol = psi4.geometry(text)
        
        psi4.core.clean()
        psi4.core.be_quiet()
        
        psi4.set_options({'basis': basis,
                          'scf_type': 'pk',
                          'reference': 'rohf',
                          'mp2_type': 'conv',
                          'e_convergence': 1e-8,
                          'd_convergence': 1e-8})

        try:
            rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
            scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
            
            A = HelperCCEnergy(mol, rhf_e, scf_wfn, freeze_core=False)

            MP2T2=A.t2start
            A.t1 = np.zeros((A.t1.shape))
            A.t2 = MP2T2
            
            MP2E = A.compute_energy(iterate=False)                    # MP2 (initial) energy
            CCSDE = A.compute_energy()                                   # exact CCSD energy

            data = pd.DataFrame(np.array([getattr(A, attr).flatten() for attr in properties]).T, columns=properties)
            data_dict[struct.split('_')[0]] = data

        except Exception as e:
            print(f"Molecule with filename {fn} failed: {e}")
            pass   

    X_test_all[basis] = np.vstack([df[top5].to_numpy() for df in data_dict.values()])
    y_test_all[basis] = np.concatenate([df["t2"].to_numpy().reshape(-1) for df in data_dict.values()])
    
    splits = {
        'X_test': X_test_all[basis],
        'y_test': y_test_all[basis]
    }
    
    joblib.dump(splits, f"out/{basis}_test_data_splits.pkl")
    t2 = time.time()


sizes = [10, 20, 40, 60, 80, 100]
X_train_all = {}
y_train_all = {}

for basis in basis_sets: 
    for n in sizes:
        filenames = train[:n]
        t1 = time.time()

        # get training molecule
        data_dict = {}
        for fn in filenames:
            struct = os.path.basename(fn)
            
            with open(fn,'r') as f:
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

            try:
                
                rhf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
                scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
                
                A=HelperCCEnergy(mol, rhf_e, scf_wfn,freeze_core=False)
    
    
                MP2T2=A.t2start
                A.t1 = np.zeros((A.t1.shape))
                A.t2 = MP2T2
                
                MP2E = A.compute_energy(iterate=False)                    # MP2 (initial) energy
                CCSDE = A.compute_energy()                                   # exact CCSD energy
    
                data=pd.DataFrame(np.array([getattr(A,attr).flatten() for attr in properties]).T,columns=properties)
                data_dict[struct.split('_')[0]]=data
            except Exception as e:
                print(f"Molecule with filename {fn} failed: {e}")
                pass   

        X_train_all[basis] = np.vstack([df[top5].to_numpy() for df in data_dict.values()])
        y_train_all[basis] = np.concatenate([df["t2"].to_numpy().reshape(-1) for df in data_dict.values()])

        splits = {
            'X_train': X_train_all[basis],
            'y_train': y_train_all[basis]
        }
        
        joblib.dump(splits, f"out/{basis}_train_data_splits_{n}.pkl")

        

        

