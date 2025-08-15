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


file_path = "out/status.txt"

if os.path.exists(file_path):
    os.remove(file_path)
    

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
        print(f"Processing {struct}")
        
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
            with open("out/failed_molecules.txt", "a") as f:
                for molecule in train:
                    f.write(f"Test molecule with filename {fn} failed: {e} \n")
            pass   

    X_test_all[basis] = np.vstack([df[top5].to_numpy() for df in data_dict.values()])
    y_test_all[basis] = np.concatenate([df["t2"].to_numpy().reshape(-1) for df in data_dict.values()])
    t2 = time.time()

    print(f"Time taken for {basis}: {t2-t1} seconds \n")
    
    with open("out/status.txt", "a") as f:
        f.write(f"(Test set), Time taken for {basis}: {t2-t1} seconds \n")


with open("out/status.txt", "a") as f:
    f.write(f"Done test set molecules, now moving onto training. \n \n")


sizes = [10, 20, 40, 60, 80, 100]
X_train_all = {}
y_train_all = {}

for basis in basis_sets: 
    for n in sizes:
        filenames = train[:n]
        t1 = time.time()
        
        print(f"{basis} Basis, N = {n} training molecules")
        print(f"Training molecules: {filenames}")

        # get training molecule
        data_dict = {}
        for fn in filenames:
            struct = os.path.basename(fn)
            print(f"Processing {struct}")
            
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


        pipeline = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('xgb', XGBRegressor(tree_method="hist", n_jobs=-1, random_state=42, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,))
        ])

  
        param_grid = {'max_depth': [10, 15, 20],
                    'n_estimators': [200, 400, 500],
                    'reg_lambda': [1e-6, 1e-3,1e-1],
                    'reg_alpha': [1e-6, 1e-3,1e-1]}
              
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            verbose=10000,
            n_jobs=-1
        )

        t1 = time.time()
        grid_search.fit(X_train_all[basis], y_train_all[basis])
        t2 = time.time()
        
        y_pred = grid_search.predict(X_test_all[basis])

        with open("out/best_params.txt", "a") as f:
            f.write(f"Best parameters for basis {basis}, N={n}: {grid_search.best_params_}\n")
        
        r2 = r2_score(y_test_all[basis], y_pred)
        mae = mean_absolute_error(y_test_all[basis], y_pred)
        rmse = root_mean_squared_error(y_test_all[basis], y_pred)

        with open("out/optimised_performance.txt", "a") as f:
            f.write(f"Basis: {basis}, N: {n}, MAE: {mae}, RMSE: {rmse}, R2: {r2}\n")
        
        joblib.dump(grid_search.best_estimator_, f"out/optimised_{basis}_best_model_{n}.pkl")

